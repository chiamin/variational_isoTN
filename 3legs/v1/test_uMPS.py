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
    #              |
    #              |    d1
    #      l  ----AL--------C---- r
    #             /        /
    #            /        /
    #          c1       c2
    psi = {
        "AL": ["s","l","c1","d1"],
        "C": ["d1","c2","r"]
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

    network.PutUniTensor("AL", AL, ["s","l","d","r"])
    network.PutUniTensor("AL'", AL.Dagger(), ["s","l","d","r"])
    network.PutUniTensor("C", C, ["l","d","r"])
    network.PutUniTensor("C'", C.Dagger(), ["l","d","r"])
    network.PutUniTensor("Sz", Sz, ["s'","s"])
    res = network.Launch()
    return res.item()

def test_gradient_descent_TFIM (Jx, hz, D, niter):
    # Define the network pattern
    # 1. Energy
    #
    #                s31                    s32
    #                  |     ul         ur    |
    #          l3 ----AL3---------C3---------AR3---- r3
    #                 /          /           /
    #            s21 / a2    c2 /       s22 / b2
    #              |/    vl    /    vr    |/
    #      l2 ----AL2---------C2---------AR2---- r2
    #             /          /           /
    #        s11 / a1    c1 /       s12 / b1
    #          |/          /          |/
    #  l1 ----AL1---------C1---------AR1---- r1
    #              wl          wr
    #
    #
    #                  d                      d
    #                  |     D          D     |
    #           D ----AL3---------C3---------AR3---- D
    #                 /          /           /
    #              d / D      D /         d / D
    #              |/     D    /     D    |/
    #       D ----AL2---------C2---------AR2---- D
    #             /          /           /
    #          d / D1    D1 /         d / D1
    #          |/          /          |/
    #  D1 ----AL1---------C1---------AR1---- D1
    #              D1          D1
    #
    allTen = {
        "AL1": ["s11","l1","wl","a1"],
        "AL2": ["s21","l2","a1","vl","a2"],
        "AL3": ["s31","l3","a2","ul"],
        "C1": ["wl","wr","c1"],
        "C2": ["vl","c1","vr","c2"],
        "C3": ["ul","c2","ur"],
        "AR1": ["s12","wr","r1","b1"],
        "AR2": ["s22","vr","b1","r2","b2"],
        "AR3": ["s32","ur","b2","r3"]
    }

    # Hx3
    #             s31                s32
    #              |    ul       ur   |
    #      l3 ----AL3-------C3-------AR3---- r3
    #             /        /         /
    #            /        /         /
    #           a2       c2        b2
    AA3x = dict()
    AA3x["AL3"] = allTen["AL3"]
    AA3x["AR3"] = allTen["AR3"]
    AA3x["C3"] = allTen["C3"]
    H3x = {"H3x": ["s31","s32","s31'","s32'"]}
    AA3x_dag = pat.Prime(AA3x, excep=["l3","a2","c2","b2","r3"])
    patEx3 = pat.Combine (AA3x, H3x, AA3x_dag)

    # Hx2
    #                 a2        c2           b2
    #            s21 /          /       s22 /
    #              |/    vl    /   vr     |/
    #      l2 ----AL2---------C2---------AR2---- r2
    #             /          /           /
    #            /          /           /
    #           a1         c1          b1
    AA2x = dict()
    AA2x["AL2"] = allTen["AL2"]
    AA2x["AR2"] = allTen["AR2"]
    AA2x["C2"] = allTen["C2"]
    AA2x["AL3"] = allTen["AL3"]
    AA2x["AR3"] = allTen["AR3"]
    AA2x["C3"] = allTen["C3"]
    H2x = {"H2x": ["s21","s22","s21'","s22'"]}
    AA2x_dag = pat.PrimeConnected(AA2x, add=["s21","s22"])
    patEx2 = pat.Combine (AA2x, H2x, AA2x_dag)

    # Hx1
    #                 a1        c1           b1
    #            s11 /          /       s12 /
    #              |/    wl    /   wr     |/
    #      l1 ----AL1---------C1---------AR1---- r1
    AA1x = dict()
    AA1x["AL1"] = allTen["AL1"]
    AA1x["AR1"] = allTen["AR1"]
    AA1x["C1"] = allTen["C1"]
    AA1x["AL2"] = allTen["AL2"]
    AA1x["AR2"] = allTen["AR2"]
    AA1x["C2"] = allTen["C2"]
    AA1x["AL3"] = allTen["AL3"]
    AA1x["AR3"] = allTen["AR3"]
    AA1x["C3"] = allTen["C3"]
    H1x = {"H1x": ["s11","s12","s11'","s12'"]}
    AA1x_dag = pat.PrimeConnected(AA1x, add=["s11","s12"])
    patEx1 = pat.Combine (AA1x, H1x, AA1x_dag)

    # Hy2
    #             s31
    #              |    ul
    #      l3 ----AL3---------C3----- ur
    #             /          /
    #        s21 / a2       / c2
    #          |/          /
    #  l2 ----AL2---------C2----- vr
    #         /    vl    /
    #        a1         c1
    AA2y = dict()
    AA2y["AL3"] = allTen["AL3"]
    AA2y["AL2"] = allTen["AL2"]
    AA2y["C3"] = allTen["C3"]
    AA2y["C2"] = allTen["C2"]
    H2y = {"H2y": ["s21","s31","s21'","s31'"]}
    AA2y_dag = pat.Prime(AA2y, excep=["l2","l3","a1","c1","ur","vr"])
    patEy2 = pat.Combine (AA2y, H2y, AA2y_dag)

    # Hy1
    #                a2         c2
    #            s21 /          /
    #              |/    vl    /
    #      l2 ----AL2---------C2----- vr
    #             /          /
    #        s11 / a1    c1 /
    #          |/          / 
    #  l1 ----AL1---------C1----- wr
    #              wl
    AA1y = dict()
    AA1y["AL1"] = allTen["AL1"]
    AA1y["AL2"] = allTen["AL2"]
    AA1y["C1"] = allTen["C1"]
    AA1y["C2"] = allTen["C2"]
    AA1y["AL3"] = allTen["AL3"]
    AA1y["C3"] = allTen["C3"]
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
        "AR1": ["s","d'","r","c2"],
        "C1'": ["l","d'","c1"]
    }
    patC1 = pat.Combine (G1L, G1R)

    # 2.2
    #
    #              c1         c2
    #          s  /          /
    #          | /          /
    #          |/    d     /
    #   l ----AL2---------C2---- r
    #         /          /
    #        /          /
    #       a1         a2
    G2L = {
        "AL2": ["s","l","a1","d","c1"],
        "C2": ["d","a2","r","c2"]
    }
    #            c1         c2
    #            /       s  /
    #           /        | /
    #          /   d'    |/
    #   l ----C2--------AR2---- r
    #        /          /
    #       /          /
    #      a1         a2
    #
    G2R = {
        "AR2": ["s","d'","a2","r","c2"],
        "C2'": ["l","a1","d'","c1"]
    }
    patC2 = pat.Combine (G2L, G2R)

    # 2.3
    #              s
    #              |
    #              |    d1
    #      l  ----AL3-------C3---- r
    #             /        /
    #            /        /
    #          c1       c2
    G3L = {
        "AL3": ["s","l","c1","d1"],
        "C3": ["d1","c2","r"]
    }
    #                      s
    #                      |
    #                 d2   |
    #      l ----C3-------AR3---- r
    #           /         /
    #          /         /
    #        c1        c2
    G3R = {
        "AR3": ["s","d2","c2","r"],
        "C3'": ["l","c1","d2"]
    }
    patC3 = pat.Combine (G3L, G3R)

    d = 2
    D1 = 2
    #D2 = 4

    net = netf.NetworkFunction()
    # Energy
    net.add(patEx3)
    net.add(patEx2)
    net.add(patEx1)
    net.add(patEy2)
    net.add(patEy1)
    # Penalty
    rho = 10
    net.add(dict(), coef=2.*rho*D1*D1)
    net.add(dict(), coef=2.*rho*D*D)
    net.add(dict(), coef=2.*rho)
    net.add(patC1, coef=-2.*rho)
    net.add(patC2, coef=-2.*rho)
    net.add(patC3, coef=-2.*rho)


    # Define tensors
    #
    #                  d                      d
    #                  |     D          D     |
    #           D ----AL3---------C3---------AR3---- D
    #                 /          /           /
    #              d / D      D /         d / D
    #              |/     D    /     D    |/
    #       D ----AL2---------C2---------AR2---- D
    #             /          /           /
    #          d / D1    D1 /         d / D1
    #          |/          /          |/
    #  D1 ----AL1---------C1---------AR1---- D1
    #              D1          D1
    #
    np.random.seed(100)
    dtype = float
    #              0
    #              | 3
    #              |/
    #       2 ----AL1---- 1
    AL1 = ut.random_isometry(d*D1, D1*D1, dtype).reshape((d,D1,D1,D1))
    #              0
    #              | 4
    #              |/
    #       1 ----AL2---- 3
    #             /
    #            2
    AL2 = ut.random_isometry(d*D*D1, D*D, dtype).reshape((d,D,D1,D,D))
    #              0
    #              |
    #       1 ----AL3---- 3
    #             /
    #            2
    AL3 = ut.random_isometry(d*D*D, D, dtype).reshape((d,D,D,D))
    #              0                      0
    #              | 3                    | 3
    #              |/                     |/
    #       2 ----AR1---- 1   =>   1 ----AR1---- 2
    AR1 = ut.random_isometry(d*D1, D1*D1, dtype).reshape((d,D1,D1,D1)).transpose((0,2,1,3))
    #              0                      0
    #              | 4                    | 4
    #              |/                     |/
    #       3 ----AR2---- 1   =>   1 ----AR2---- 3
    #             /                      /
    #            2                      2
    AR2 = ut.random_isometry(d*D*D1, D*D, dtype).reshape((d,D,D1,D,D)).transpose((0,3,2,1,4))
    #              0                      0
    #              |                      |
    #       3 ----AR3---- 1   =>   1 ----AR3---- 3
    #             /                      /
    #            2                      2
    AR3 = ut.random_isometry(d*D*D, D, dtype).reshape((d,D,D,D)).transpose((0,3,2,1))
    C1 = ut.random_isometry(D1*D1, D1, dtype).reshape((D1,D1,D1))
    C2 = ut.random_isometry(D*D1*D, D, dtype).reshape((D,D1,D,D))
    C3 = ut.random_tensor((D,D,D), dtype)
    C3 /= np.linalg.norm(C3)

    # Define Hamiltonian
    Hx = hamilt.make_Hx_TFIM2(Jx, hz)
    Hy = hamilt.make_Hy_TFIM2(Jx, hz)

    # To UniTensors
    Hx = uniten.ToUniTensor (Hx, ["s1","s2","s1'","s2'"])
    Hy = uniten.ToUniTensor (Hy, ["s1","s2","s1'","s2'"])
    AL1 = uniten.ToUniTensor (AL1, ["s","l","r","u"])
    AL2 = uniten.ToUniTensor (AL2, ["s","l","d","r","u"])
    AL3 = uniten.ToUniTensor (AL3, ["s","l","d","r"])
    AR1 = uniten.ToUniTensor (AR1, ["s","l","r","u"])
    AR2 = uniten.ToUniTensor (AR2, ["s","l","d","r","u"])
    AR3 = uniten.ToUniTensor (AR3, ["s","l","d","r"])
    C1 = uniten.ToUniTensor (C1, ["l","r","u"])
    C2 = uniten.ToUniTensor (C2, ["l","d","r","u"])
    C3 = uniten.ToUniTensor (C3, ["l","d","r"])


    # Define the cost function
    cost_func = cf.CostFunction(net)

    # Put tensors
    cost_func.nets.putTensor("H1x", Hx, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("H2x", Hx, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("H3x", Hx, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("H1y", Hy, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("H2y", Hy, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("AL1", AL1, ["s","l","r","u"])
    cost_func.nets.putTensor("AL2", AL2, ["s","l","d","r","u"])
    cost_func.nets.putTensor("AL3", AL3, ["s","l","d","r"])
    cost_func.nets.putTensor("AR1", AR1, ["s","l","r","u"])
    cost_func.nets.putTensor("AR2", AR2, ["s","l","d","r","u"])
    cost_func.nets.putTensor("AR3", AR3, ["s","l","d","r"])
    cost_func.nets.putTensor("C1", C1, ["l","r","u"])
    cost_func.nets.putTensor("C2", C2, ["l","d","r","u"])
    cost_func.nets.putTensor("C3", C3, ["l","d","r"])
    cost_func.nets.putTensor("AL1'", AL1.Dagger(), ["s","l","r","u"])
    cost_func.nets.putTensor("AL2'", AL2.Dagger(), ["s","l","d","r","u"])
    cost_func.nets.putTensor("AL3'", AL3.Dagger(), ["s","l","d","r"])
    cost_func.nets.putTensor("AR1'", AR1.Dagger(), ["s","l","r","u"])
    cost_func.nets.putTensor("AR2'", AR2.Dagger(), ["s","l","d","r","u"])
    cost_func.nets.putTensor("AR3'", AR3.Dagger(), ["s","l","d","r"])
    cost_func.nets.putTensor("C1'", C1.Dagger(), ["l","r","u"])
    cost_func.nets.putTensor("C2'", C2.Dagger(), ["l","d","r","u"])
    cost_func.nets.putTensor("C3'", C3.Dagger(), ["l","d","r"])

    #exact = -0.317878160486075589

    constraint_crit = 1e-4
    rho_max = 200
    nGD = 1
    N_linesearch = 20
    for i in range(niter):
        cost_func.setGarget("AL1")
        AL1, value, slope = gd.gradient_descent (cost_func, AL1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","l"])

        cost_func.setGarget("AL2")
        AL2, value, slope = gd.gradient_descent (cost_func, AL2, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","l","d"])

        cost_func.setGarget("AL3")
        AL3, value, slope = gd.gradient_descent (cost_func, AL3, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","l","d"])

        cost_func.setGarget("AR1")
        AR1, value, slope = gd.gradient_descent (cost_func, AR1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r"])

        cost_func.setGarget("AR2")
        AR2, value, slope = gd.gradient_descent (cost_func, AR2, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r","d"])

        cost_func.setGarget("AR3")
        AR3, value, slope = gd.gradient_descent (cost_func, AR3, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r","d"])

        cost_func.setGarget("C1")
        C1, value, slope = gd.gradient_descent (cost_func, C1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","r"])

        cost_func.setGarget("C2")
        C2, value, slope = gd.gradient_descent (cost_func, C2, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","d","r"])

        cost_func.setGarget("C3")
        C3, value, slope = gd.gradient_descent (cost_func, C3, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen normalize")

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
            en = sum(cost_func.nets.terms[:5])/3
            # Measure Sz
            sz = measureSz(AL3, C3)
            print(i, en, sz, slope)
            print("\t\t",cost_func.nets.terms[:5])
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
    Jx = -1
    D = 4
    niter = 1000

    hzs = [-1]
    szs = []
    for hz in hzs:
        sz = test_gradient_descent_TFIM (Jx, hz, D, niter)
        #szs.append(sz)

    #data = np.column_stack((hzs, szs))
    #np.savetxt("output.txt", data)

    #plt.plot (hzs, szs, marker='.')
    #plt.show()
