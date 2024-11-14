import os, sys, copy
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0,'/home/chiamin/Cytnx/Install/')
import cytnx
import numpy as np
sys.path.append('../../')
import Pattern as pat
from ncon import ncon
import NetworkFunction as netf
import Hamiltonian as hamilt
import UniTensorTools as uniten
import Utility as ut
import CostFunction as cf
import GradientDesent as gd
import matplotlib.pyplot as plt

def measureSz(AL, C):
    #              s
    #              | u1       u2
    #              | /        /
    #              |/   d1   /
    #      l  ----AL--------C---- r
    #             /        /
    #            /        /
    #          c1       c2
    psi = {
        "AL": ["s","l","c1","u1","d1"],
        "C": ["d1","c2","u2","r"]
    }
    #   s'
    #   |
    #   Sz
    #   |
    #   s
    Sz = {"Sz": ["s'","s"]}

    psidag = pat.PrimeConnected(psi, add=["s"])

    p = pat.Combine (psi, Sz, psidag)

    network = cytnx.Network()
    network.FromString (pat.ToNetworkString(p))


    Sz = hamilt.make_Sz()
    Sz = uniten.ToUniTensor (Sz, ["s'","s"])

    network.PutUniTensor("AL", AL, ["s","l","d","u","r"])
    network.PutUniTensor("AL'", AL.Dagger(), ["s","l","d","u","r"])
    network.PutUniTensor("C", C, ["l","d","r","u"])
    network.PutUniTensor("C'", C.Dagger(), ["l","d","r","u"])
    network.PutUniTensor("Sz", Sz, ["s'","s"])
    res = network.Launch()
    return res.item()

def test_gradient_descent_TFIM (J, D, niter):
    # Define the network pattern
    # 1. Energy
    #
    #                s31                    s32
    #                  |     ul         ur    |
    #          l3 ----AL1---------C1---------AR1---- r3
    #                 /          /           /
    #            s21 / a2    c2 /       s22 / b2
    #              |/    vl    /    vr    |/
    #      l2 ----AL----------C----------AR---- r2
    #             /          /           /
    #        s11 / a1    c1 /       s12 / b1
    #          |/          /          |/
    #  l1 ----AL1---------C1---------AR1---- r1
    #              wl          wr
    #
    allTen = {
        "AL1": ["s11","l1","wl","a1"],
        "AL": ["s21","l2","a1","a2","vl"],
        "C1": ["wl","wr","c1"],
        "C": ["vl","c1","vr","c2"],
        "AR1": ["s12","r1","wr","b1"],
        "AR": ["s22","r2","b1","b2","vr"],
    }

    # Hx1
    #                 a1        c1           b1
    #            s11 /          /       s12 /
    #              |/    wl    /   wr     |/
    #      l1 ----AL1---------C1---------AR1---- r1
    AA1x = dict()
    AA1x["AL1"] = allTen["AL1"]
    AA1x["AR1"] = allTen["AR1"]
    AA1x["C1"] = allTen["C1"]
    AA1x["AL"] = allTen["AL"]
    AA1x["AR"] = allTen["AR"]
    AA1x["C"] = allTen["C"]
    H1x = {"H1x": ["s11","s12","s11'","s12'"]}
    AA1x_dag = pat.PrimeConnected(AA1x, add=["s11","s12"])
    patEx1 = pat.Combine (AA1x, H1x, AA1x_dag)

    # Hx2
    #                 a2        c2           b2
    #            s21 /          /       s22 /
    #              |/    vl    /   vr     |/
    #      l2 ----AL2---------C2---------AR2---- r2
    #             /          /           /
    #            /          /           /
    #           a1         c1          b1
    AA2x = dict()
    AA2x["AL"] = allTen["AL"]
    AA2x["AR"] = allTen["AR"]
    AA2x["C"] = allTen["C"]
    H2x = {"H2x": ["s21","s22","s21'","s22'"]}
    AA2x_dag = pat.PrimeConnected(AA2x, add=["s21","s22"])
    patEx2 = pat.Combine (AA2x, H2x, AA2x_dag)

    # Hy
    #                a2         c2
    #            s21 /          /
    #              |/    vl    /
    #      l2 ----AL----------C----- vr
    #             /          /
    #        s11 / a1    c1 /
    #          |/          / 
    #  l1 ----AL1---------C1----- wr
    #              wl
    AA1y = dict()
    AA1y["AL1"] = allTen["AL1"]
    AA1y["AL"] = allTen["AL"]
    AA1y["C1"] = allTen["C1"]
    AA1y["C"] = allTen["C"]
    H1y = {"H1y": ["s11","s21","s11'","s21'"]}
    AA1y_dag = pat.PrimeConnected(AA1y, add=["s11","s21"])
    patEy1 = pat.Combine (AA1y, H1y, AA1y_dag)

    # 2. Constraint penalty
    # 2.1
    #
    #              c1       c2
    #           s  /        /
    #           | /        /
    #           |/   d    /
    #    l ----AL1-------C1---- r
    G1L = {
        "AL1": ["s","l","d","c1"],
        "C1": ["d","r","c2"]
    }
    #            c1        c2
    #            /      s  /
    #           /       | /
    #          /        |/
    #   l ----C1-------AR1---- r
    #              d'
    G1R = {
        "AR1": ["s","r","d'","c2"],
        "C1'": ["l","d'","c1"]
    }
    patC1 = pat.Combine (G1L, G1R)

    # 2.2
    #
    #              c1         c2
    #          s  /          /
    #          | /          /
    #          |/    d     /
    #   l ----AL----------C---- r
    #         /          /
    #        /          /
    #       a1         a2
    G2L = {
        "AL": ["s","l","a1","c1","d"],
        "C": ["d","a2","r","c2"]
    }
    #            c1         c2
    #            /       s  /
    #           /        | /
    #          /   d'    |/
    #   l ----C---------AR---- r
    #        /          /
    #       /          /
    #      a1         a2
    #
    G2R = {
        "AR": ["s","r","a2","c2","d'"],
        "C'": ["l","a1","d'","c1"]
    }
    patC2 = pat.Combine (G2L, G2R)

    d = 2
    D1 = 2
    #D2 = 4

    net = netf.NetworkFunction()
    # Energy
    net.add(patEx1)
    net.add(patEx2)
    net.add(patEy1)
    # Penalty
    rho = 10
    net.add(dict(), coef=2.*rho*D1*D1)
    net.add(dict(), coef=2.*rho)
    net.add(patC1, coef=-2.*rho)
    net.add(patC2, coef=-2.*rho)


    # Define tensors
    #
    #                  d                      d
    #                  |     D1         D1    |
    #          D1 ----AL1---------C1---------AR1---- D1
    #                 /          /           /
    #              d / D1       / D1      d / D1
    #              |/     D    /     D    |/
    #       D ----AL----------C----------AR---- D
    #             /          /           /
    #          d / D1       / D1      d / D1
    #          |/          /          |/
    #  D1 ----AL1---------C1---------AR1---- D1
    #              D1          D1
    #
    np.random.seed(100)
    dtype = float
    #              0
    #              | 3
    #              |/
    #       1 ----AL1---- 2
    AL1 = ut.random_isometry(d*D1, D1*D1, dtype).reshape((d,D1,D1,D1))
    #              0
    #              | 3
    #              |/
    #       1 ----AL---- 4
    #             /
    #            2
    AL = ut.random_isometry(d*D*D1*D1, D, dtype).reshape((d,D,D1,D1,D))
    #              0
    #              | 3
    #              |/
    #       2 ----AR1---- 1
    AR1 = ut.random_isometry(d*D1, D1*D1, dtype).reshape((d,D1,D1,D1))
    #              0
    #              | 3
    #              |/
    #       4 ----AR2---- 1
    #             /
    #            2
    AR = ut.random_isometry(d*D*D1*D1, D, dtype).reshape((d,D,D1,D1,D))
    #                2
    #               /
    #        0 ----C1---- 1
    C1 = ut.random_isometry(D1*D1, D1, dtype).reshape((D1,D1,D1))
    #                3
    #               /
    #       0 ----C2---- 2
    #             /
    #            1
    C = ut.random_tensor((D,D1,D,D1), dtype)
    C /= np.linalg.norm(C)

    # Define Hamiltonian
    H = hamilt.make_H_Heisenberg(J)

    # To UniTensors
    H = uniten.ToUniTensor (H, ["s1","s2","s1'","s2'"])
    AL1 = uniten.ToUniTensor (AL1, ["s","l","r","u"])
    AL = uniten.ToUniTensor (AL, ["s","l","d","u","r"])
    AR1 = uniten.ToUniTensor (AR1, ["s","r","l","u"])
    AR = uniten.ToUniTensor (AR, ["s","r","d","u","l"])
    C1 = uniten.ToUniTensor (C1, ["l","r","u"])
    C = uniten.ToUniTensor (C, ["l","d","r","u"])


    # Define the cost function
    cost_func = cf.CostFunction(net)

    # Put tensors
    cost_func.nets.putTensor("H1x",  H,            H.labels())
    cost_func.nets.putTensor("H2x",  H,            H.labels())
    cost_func.nets.putTensor("H1y",  H,            H.labels())
    cost_func.nets.putTensor("AL1",  AL1,          AL1.labels())
    cost_func.nets.putTensor("AL",   AL,           AL.labels())
    cost_func.nets.putTensor("AR1",  AR1,          AR1.labels())
    cost_func.nets.putTensor("AR",   AR,           AR.labels())
    cost_func.nets.putTensor("C1",   C1,           C1.labels())
    cost_func.nets.putTensor("C",    C,            C.labels())
    cost_func.nets.putTensor("AL1'", AL1.Dagger(), AL1.labels())
    cost_func.nets.putTensor("AL'",  AL.Dagger(),  AL.labels())
    cost_func.nets.putTensor("AR1'", AR1.Dagger(), AR1.labels())
    cost_func.nets.putTensor("AR'",  AR.Dagger(),  AR.labels())
    cost_func.nets.putTensor("C1'",  C1.Dagger(),  C1.labels())
    cost_func.nets.putTensor("C'",   C.Dagger(),   C.labels())

    #exact = -0.317878160486075589

    constraint_crit = 1e-4
    rho_max = 200
    nGD = 1
    N_linesearch = 20
    for i in range(niter):
        cost_func.setGarget("AL1")
        AL1, value, slope = gd.gradient_descent (cost_func, AL1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","l"])

        cost_func.setGarget("AL")
        AL, value, slope = gd.gradient_descent (cost_func, AL, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","l","d","u"])

        cost_func.setGarget("AR1")
        AR1, value, slope = gd.gradient_descent (cost_func, AR1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r"])

        cost_func.setGarget("AR")
        AR, value, slope = gd.gradient_descent (cost_func, AR, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r","d","u"])

        cost_func.setGarget("C1")
        C1, value, slope = gd.gradient_descent (cost_func, C1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","r"])

        cost_func.setGarget("C")
        C, value, slope = gd.gradient_descent (cost_func, C, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen normalize")

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
        #if rho < rho_max:
        #    rho *= 1.2

        if i % 1 == 0:
            constraint_weight = sum(cost_func.nets.terms[-4:])
            en_x1, en_x2, en_y = cost_func.nets.terms[:3]
            en = ((en_x1+en_x2)*(3/2) + 2*en_y)/3
            # Measure Sz
            sz = measureSz(AL, C)
            print(i, en, sz, slope)
            print("\t\t",cost_func.nets.terms[:3])
            #print(cost_func.nets.terms[:5], constraint_weight, slope, rho)
            #print(i, cost_func.nets.terms[0].item().real, penalty, slope)

    return sz

def plotfig (fname):
    data = np.loadtxt(fname)
    hzs = data[:,0]
    szs = data[:,1]
    plt.plot (-hzs, 0.5*szs, marker='.')
    plt.show()

if __name__ == '__main__':
    J = -1
    D = 2
    niter = 1000

    sz = test_gradient_descent_TFIM (J, D, niter)

