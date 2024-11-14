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

def expandTensor(T, labels, shape):
    T.permute(labels)

    ToNpArray

def getDiff (AL, C, AR):
    #
    #             c1       c2
    #          s  /        /
    #          | /        /
    #          |/        /
    #   l ----AL--------C---- r
    #               d
    G1 = {
        "AL": ["l","s","d","c1"],
        "C": ["d","r","c2"],
        "TOUT": ["l","s","r","c1","c2"]
    }
    #            c1        c2
    #            /      s  /
    #           /       | /
    #          /        |/
    #   l ----C--------AR---- r
    #              d'
    G2 = {
        "AR": ["d'","s","r","c2"],
        "C": ["l","d'","c1"],
        "TOUT": ["l","s","r","c1","c2"]
    }

    #
    network1 = cytnx.Network()
    network1.FromString (pat.ToNetworkString(G1))
    network2 = cytnx.Network()
    network2.FromString (pat.ToNetworkString(G2))

    network1.PutUniTensor("AL", AL, ['l','s',"r","c"])
    network1.PutUniTensor("C", C, ['l',"r","c"])
    T1 = network1.Launch()
    network2.PutUniTensor("AR", AR, ['l','s',"r","c"])
    network2.PutUniTensor("C", C, ['l',"r","c"])
    T2 = network2.Launch()

    T = T1 - T2
    return T.Norm().item()

def test_gradient_descent_TFIM():
    # Define the network pattern
    net = netf.NetworkFunction()
    # 1. Energy
    #             s21                s22
    #              |    u1       u2   |
    #      l2 ----AL2-------C2-------AR2---- r2
    #             /        /         /
    #        s11 / c1   c /     s12 / c2
    #          |/        /        |/
    #  l1 ----AL1-------C1-------AR1---- r1
    #              d1       d2
    #
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
    #  D1 <= d

    #              s1                 s2
    #              |    u1       u2   |
    #       l ----AL2-------C2-------AR2---- r
    #             /        /         /
    #            /        /         /
    #          c1        c        c2
    pattern_AAx = {
        "AL2": ["l","s1","u1","c1"],
        "AR2": ["u2","s2","r","c2"],
        "C2": ["u1","u2","c"]
    }
    pattern_Hx = {"Hx": ["s1","s2","s1'","s2'"]}
    pattern_AAxdag = pat.Prime(pattern_AAx, excep=["l","c1","c","c2","r"])
    pattern_Ex = pat.Combine (pattern_AAx, pattern_Hx, pattern_AAxdag)

    pattern_AA = {
        "AL1": ["l1","s11","d1","c1"],
        "AR1": ["d2","s12","r1","c2"],
        "AL2": ["l2","s21","u1","c1"],
        "AR2": ["u2","s22","r2","c2"],
        "C1": ["d1","d2","c"],
        "C2": ["u1","u2","c"]
    }
    pattern_Hx2 = {"Hx2": ["s11","s12","s11'","s12'"]}
    pattern_AAdag = pat.PrimeConnected(pattern_AA, add=["s11","s12"])
    pattern_Ex2 = pat.Combine (pattern_AA, pattern_Hx2, pattern_AAdag)
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
    D2 = 2

    # Energy
    net.add(pattern_Ex)
    net.add(pattern_Ex2)
    net.add(pattern_Ey)
    # Penalty
    rho = 10
    net.add(dict(), coef=2.*rho*D1*D1)
    net.add(pattern_C1, coef=-2.*rho)
    net.add(dict(), coef=2.*rho)
    net.add(pattern_C2, coef=-2.*rho)


    # Define Hamiltonian
    Jx = -0.
    hz = -1.
    Hx = hamilt.make_H_TFIM2(Jx, hz)
    Hy = hamilt.make_H_TFIM2(Jx, hz)

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
    #              1                      1
    #              | 3                    | 3
    #              |/                     |/
    #       2 ----AR1---- 0   =>   0 ----AR1---- 2
    AR1 = ut.random_isometry(d*D1, D1*D1, dtype).reshape((D1,d,D1,D1)).transpose((2,1,0,3))
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
    Hx = uniten.ToUniTensor (Hx, ["s1","s2","s1'","s2'"])
    Hy = uniten.ToUniTensor (Hy, ["s1","s2","s1'","s2'"])
    AL1 = uniten.ToUniTensor (AL1, ["l","s","r","c"])
    AL2 = uniten.ToUniTensor (AL2, ["l","s","r","c"])
    AR1 = uniten.ToUniTensor (AR1, ["l","s","r","c"])
    AR2 = uniten.ToUniTensor (AR2, ["l","s","r","c"])
    C1 = uniten.ToUniTensor (C1, ["l","r","c"])
    C2 = uniten.ToUniTensor (C2, ["l","r","c"])


    # Define the cost function
    cost_func = cf.CostFunction(net)

    # Put tensors
    cost_func.nets.putTensor("Hx", Hx, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("Hx2", Hx, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("Hy", Hy, ["s1","s2","s1'","s2'"])
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
    rho_max = 2000
    nGD = 1
    N_linesearch = 40
    N_each = 1
    for i in range(4001):
        for j in range(N_each):
            cost_func.setGarget("AL1")
            AL1, value, slope = gd.gradient_descent (cost_func, AL1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","s"])

        for j in range(N_each):
            cost_func.setGarget("AL2")
            AL2, value, slope = gd.gradient_descent (cost_func, AL2, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","s","c"])

        for j in range(N_each):
            cost_func.setGarget("AR1")
            AR1, value, slope = gd.gradient_descent (cost_func, AR1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r"])

        for j in range(N_each):
            cost_func.setGarget("AR2")
            AR2, value, slope = gd.gradient_descent (cost_func, AR2, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r","c"])

        for j in range(N_each):
            cost_func.setGarget("C2")
            C2, value, slope = gd.gradient_descent (cost_func, C2, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen normalize")

        for j in range(N_each):
            cost_func.setGarget("C1")
            C1, value, slope = gd.gradient_descent (cost_func, C1, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","r"])


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
            rho *= 1.1

        if i % 1 == 0:
            constraint1 = getDiff(AL1, C1, AR1)
            constraint2 = getDiff(AL2, C2, AR2)
            en = sum(cost_func.nets.terms[:3])/2
            print(i, en, cost_func.nets.terms[:3], constraint1, constraint2)
            #print(i, cost_func.nets.terms[0].item().real, penalty, slope)

test_gradient_descent_TFIM()
