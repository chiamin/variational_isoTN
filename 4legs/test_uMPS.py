import os, sys, copy
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0,'/home/chiamin/Cytnx/Install/')
import cytnx
import numpy as np
sys.path.append('../')
import Pattern as pat
from ncon import ncon
import NetworkFunction as netf
import Hamiltonian as hamilt
import UniTensorTools as uniten
import Utility as ut
import CostFunction as cf
import GradientDesent as gd
import matplotlib.pyplot as plt

def measureSz(AL2, C2):
    #              s
    #              |
    #              |    d1
    #      l  ----AL2-------C2---- r
    #             /        /
    #            /        /
    #          c1       c2
    psi = {
        "AL2": ["l","s","d1","c1"],
        "C2": ["d1","r","c2"]
    }
    #   s'
    #   |
    #   Sz
    #   |
    #   s
    Sz = {"Sz": ["s'","s"]}

    psidag = pat.Prime(psi, only=["s"])

    p = pat.Combine (psi, Sz, psidag)

    network = cytnx.Network()
    network.FromString (pat.ToNetworkString(p))


    Sz = hamilt.make_Sz()
    Sz = uniten.ToUniTensor (Sz, ["s'","s"])

    network.PutUniTensor("AL2", AL2, ['l','s',"r","c"])
    network.PutUniTensor("AL2'", AL2.Dagger(), ['l','s',"r","c"])
    network.PutUniTensor("C2", C2, ['l',"r","c"])
    network.PutUniTensor("C2'", C2.Dagger(), ['l',"r","c"])
    network.PutUniTensor("Sz", Sz, ["s'","s"])
    res = network.Launch()
    return res.item()

def test_gradient_descent_TFIM (Jx, hz, D2, niter):
    # Define the network pattern
    # 1. Energy
    #                     s41                    s42
    #                      |    ul          ur    |
    #              l4 ----AL4---------C4---------AR4---- r4
    #                     /          /           /
    #                s31 / a3    c3 /       s32 / b3
    #                  |/          /          |/
    #          l3 ----AL3---------C3---------AR3---- r3
    #                 /          /           /
    #            s21 / a2    c2 /       s22 / b2
    #              |/          /          |/
    #      l2 ----AL2---------C2---------AR2---- r2
    #             /          /           /
    #        s11 / a1    c1 /       s12 / b1
    #          |/          /          |/
    #  l1 ----AL1---------C1---------AR1---- r1
    #              tl          tr
    #
    #
    #             s41                s42
    #              |    ul       ur   |
    #      l4 ----AL4-------C4-------AR4---- r4
    #             /        /         /
    #            /        /         /
    #           a3       c3        b3
    AA4x = {
        "AL4": ["l4","s41","ul","a3"],
        "AR4": ["ur","s42","r4","b3"],
        "C4": ["ul","ur","c3"]
    }
    H4x = {"Hx": ["s41","s42","s41'","s42'"]}
    AA4x_dag = pat.Prime(AA4x, excep=["l4","a3","c3","b3","r4"])
    pEx = pat.Combine (AA4x, H4x, AA4x_dag)

    #             s2
    #              |    d2
    #      l2 ----AL2-------C2----- r2
    #             /        /
    #         s1 / c1     / c2
    #          |/        /
    #  l1 ----AL1-------C1----- r1
    #              d1
    pattern_AAy = {
        "AL1": ["l1","s1","d1","c1"],
        "AL2": ["l2","s2","d2","c1"],
        "C1": ["d1","r1","c2"],
        "C2": ["d2","r2","c2"]
    }
    pattern_Hy = {"Hy": ["s1","s2","s1'","s2'"]}
    pattern_AAydag = pat.Prime(pattern_AAy, excep=["l1","l2","r1","r2"])
    pattern_Ey = pat.Combine (pattern_AAy, pattern_Hy, pattern_AAydag)

    # 2. Constraint penalty
    # 2.1
    #
    #             c1       c2
    #          s  /        /
    #          | /        /
    #          |/        /
    #   l ----AL1-------C1---- r
    #               d
    pattern_G1 = {
        "AL1": ["l","s","d","c1"],
        "C1": ["d","r","c2"]
    }
    #            c1        c2
    #            /      s  /
    #           /       | /
    #          /        |/
    #   l ----C1-------AR1---- r
    #              d'
    pattern_G2 = {
        "AR1": ["d'","s","r","c2"],
        "C1'": ["l","d'","c1"]
    }
    pattern_C1 = pat.Combine (pattern_G1, pattern_G2)

    # 2.2
    #              s
    #              |
    #              |    d1
    #      l  ----AL2-------C2---- r
    #             /        /
    #            /        /
    #          c1       c2
    pattern_K1 = {
        "AL2": ["l","s","d1","c1"],
        "C2": ["d1","r","c2"]
    }
    #                      s
    #                      |
    #                 d2   |
    #      l ----C2-------AR2---- r
    #           /         /
    #          /         /
    #        c1        c2
    pattern_K2 = {
        "AR2": ["d2","s","r","c2"],
        "C2'": ["l","d2","c1"]
    }
    pattern_C2 = pat.Combine (pattern_K1, pattern_K2)

    d = 2
    D1 = 2
    #D2 = 4

    net = netf.NetworkFunction()
    # Energy
    net.add(pattern_Ex)
    net.add(pattern_Ey)
    # Penalty
    rho = 10
    net.add(dict(), coef=2.*rho*D1*D1)
    net.add(pattern_C1, coef=-2.*rho)
    net.add(dict(), coef=2.*rho)
    net.add(pattern_C2, coef=-2.*rho)


    # Define Hamiltonian
    H = hamilt.make_H_TFIM2(Jx, hz)

    # Define tensors
    #
    #              d                  d
    #              |    D2       D2   |
    #      D2 ----AL2-------C2-------AR2---- D2
    #             /        /         /
    #          d / D1     / D1    d / D1
    #          |/        /        |/
    #  D1 ----AL1-------C1-------AR1---- D1
    #              D1       D1
    #
    np.random.seed(100)
    dtype = float
    AL1 = ut.random_isometry(d*D1, D1*D1, dtype).reshape((D1,d,D1,D1))
    AR1 = ut.random_isometry(d*D1, D1*D1, dtype).reshape((D1,d,D1,D1))
    #              1                      1
    #              |                      |
    #       0 ----AL2---- 3   =>   0 ----AL2---- 2
    #             /                      /
    #            2                      3
    AL2 = ut.random_isometry(d*D1*D2, D2, dtype).reshape((D2,d,D1,D2)).transpose((0,1,3,2))
    #              1                      1
    #              |                      |
    #       3 ----AR2---- 0   =>   0 ----AR2---- 2
    #             /                      /
    #            2                      3
    AR2 = ut.random_isometry(d*D1*D2, D2, dtype).reshape((D2,d,D1,D2)).transpose((3,1,0,2))
    C1 = ut.random_isometry(D1*D1, D1, dtype).reshape((D1,D1,D1))
    C2 = ut.random_tensor((D2,D2,D1), dtype)
    C2 /= np.linalg.norm(C2)

    # To UniTensors
    H = uniten.ToUniTensor (H, ["s1","s2","s1'","s2'"])
    AL1 = uniten.ToUniTensor (AL1, ["l","s","r","c"])
    AL2 = uniten.ToUniTensor (AL2, ["l","s","r","c"])
    AR1 = uniten.ToUniTensor (AR1, ["l","s","r","c"])
    AR2 = uniten.ToUniTensor (AR2, ["l","s","r","c"])
    C1 = uniten.ToUniTensor (C1, ["l","r","c"])
    C2 = uniten.ToUniTensor (C2, ["l","r","c"])


    # Define the cost function
    cost_func = cf.CostFunction(net)

    # Put tensors
    cost_func.nets.putTensor("Hx", H, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("Hy", H, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("AL1", AL1, ['l','s',"r","c"])
    cost_func.nets.putTensor("AL2", AL2, ['l','s',"r","c"])
    cost_func.nets.putTensor("AR1", AR1, ['l','s',"r","c"])
    cost_func.nets.putTensor("AR2", AR2, ['l','s',"r","c"])
    cost_func.nets.putTensor("C1", C1, ['l',"r","c"])
    cost_func.nets.putTensor("C2", C2, ['l',"r","c"])
    cost_func.nets.putTensor("AL1'", AL1.Dagger(), ['l','s',"r","c"])
    cost_func.nets.putTensor("AL2'", AL2.Dagger(), ['l','s',"r","c"])
    cost_func.nets.putTensor("AR1'", AR1.Dagger(), ['l','s',"r","c"])
    cost_func.nets.putTensor("AR2'", AR2.Dagger(), ['l','s',"r","c"])
    cost_func.nets.putTensor("C1'", C1.Dagger(), ['l',"r","c"])
    cost_func.nets.putTensor("C2'", C2.Dagger(), ['l',"r","c"])

    #exact = -0.317878160486075589

    constraint_crit = 1e-4
    rho_max = 200
    nGD = 1
    N_linesearch = 20
    for i in range(niter):
        cost_func.setGarget("AL1")
        AL1, value, slope = gd.gradient_descent (cost_func, AL1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","s"])

        cost_func.setGarget("AL2")
        AL2, value, slope = gd.gradient_descent (cost_func, AL2, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","s","c"])

        cost_func.setGarget("AR1")
        AR1, value, slope = gd.gradient_descent (cost_func, AR1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r"])

        cost_func.setGarget("AR2")
        AR2, value, slope = gd.gradient_descent (cost_func, AR2, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r","c"])

        cost_func.setGarget("C1")
        C1, value, slope = gd.gradient_descent (cost_func, C1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","r"])

        cost_func.setGarget("C2")
        C2, value, slope = gd.gradient_descent (cost_func, C2, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen normalize")

        # Update Lambda
        '''g = cost_func.nets.gradient("Lambda")
        Lambda = Lambda + rho*g
        Lambda.relabels_(g.labels())

        # Update rho
        constraint_violation = g.Norm().item()
        if constraint_violation > constraint_crit:
            rho = min(2*rho, rho_max)
            cost_func.nets.networks[3][0] = 2*rho
            cost_func.nets.networks[4][0] = -2*rho'''
        if rho < rho_max:
            rho *= 1.2

        if i % 1 == 0:
            constraint_weight = sum(cost_func.nets.terms[-4:])
            en = sum(cost_func.nets.terms[:2])/2
            print(i, en, cost_func.nets.terms[:2], constraint_weight, slope, rho)
            #print(i, cost_func.nets.terms[0].item().real, penalty, slope)

    # Measure Sz
    sz = measureSz(AL2, C2)
    return sz

def plotfig (fname):
    data = np.loadtxt(fname)
    hzs = data[:,0]
    szs = data[:,1]
    plt.plot (-hzs, 0.5*szs, marker='.')
    plt.show()

if __name__ == '__main__':
    plotfig("output.txt")
    exit()

    Jx = -1
    D2 = 4
    niter = 1000

    hzs = -np.arange(0,3,0.2)
    szs = []
    for hz in hzs:
        sz = test_gradient_descent_TFIM (Jx, hz, D2, niter)
        szs.append(sz)

    data = np.column_stack((hzs, szs))
    np.savetxt("output.txt", data)

    plt.plot (hzs, szs, marker='.')
    plt.show()
