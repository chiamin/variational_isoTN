import os, sys, copy
os.environ["OMP_NUM_THREADS"] = "1"
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

def getDiff (AL, C, AR):
    #
    #          s
    #          |
    #          |
    #   l ----AL--------C---- r
    #               c
    G1 = {
        "AL": ["l", "s", "c"],
        "C": ["c", "r"],
        "TOUT": ["l","s","r"]
    }
    #                   s
    #                   |
    #                   |
    #   l ----C--------AR---- r
    #              c
    G2 = {
        "AR": ["c","s","r"],
        "C": ["l","c"],
        "TOUT": ["l","s","r"]
    }
    #
    network1 = cytnx.Network()
    network1.FromString (pat.ToNetworkString(G1))
    network2 = cytnx.Network()
    network2.FromString (pat.ToNetworkString(G2))

    network1.PutUniTensor("AL", AL, ['l','s',"r"])
    network1.PutUniTensor("C", C, ['l',"r"])
    T1 = network1.Launch()
    network2.PutUniTensor("AR", AR, ['l','s',"r"])
    network2.PutUniTensor("C", C, ['l',"r"])
    T2 = network2.Launch()

    T = T1 - T2
    return T.Norm().item()

def test_gradient_descent_TFIM():
    rho = 10

    # Define the network pattern
    net = netf.NetworkFunction()
    # 1. Energy
    #    -----AL'----C'----AR'---
    #    |    |            |    |
    #    |    |____________|    |
    #    |   (______________)   |
    #    |    |            |    |
    #    |    |            |    |
    #    -----AL-----C-----AR----
    pattern_AA = {
        "AL": ["l", "s1", "cl"],
        "C": ["cl", "cr"],
        "AR": ["cr", "s2", "r"],
    }
    pattern_AAdag = pat.Prime(pattern_AA)
    pattern_AAdag = pat.ReplaceIndex (pattern_AAdag, ["l'","r'"], ["l","r"])

    pattern_en = pat.Combine (pattern_AA, pattern_AAdag)
    pattern_en["H"] = ["s1","s2","s1'","s2'"]

    # 2. Lagrangian multiplier
    #    ---Lambda--------
    #    |    |          |
    #  l |  s |          | r
    #    |    |   c      |
    #    -----AL------C---
    pattern_L1 = {
        "AL": ["l", "s", "c"],
        "C": ["c", "r"],
        "Lambda": ["l","s","r"]
    }
    #    ---------Lambda----
    #    |           |     |
    #  l |           | s   | r
    #    |       c   |     |
    #    -----C------AR-----
    pattern_L2 = {
        "AR": ["c", "s", "r"],
        "C": ["l", "c"],
        "Lambda": ["l","s","r"]
    }

    # 3. Constraint penalty
    #    ---C'-----AR----------
    #    |     2   |          |
    #  1 |         | 3        | 4
    #    |         |    5     |
    #    ----------AL------C---
    pattern_penalty = {
        "AL": ["1", "3", "5"],
        "C": ["5","4"],
        "AR": ["2","3","4"],
        "C'": ["1","2"],
    }

    # Energy
    net.add(pattern_en)
    # Lagrange multiplier
    #net.add(pattern_L1)
    #net.add(pattern_L2, coef=-1.)
    # Penalty
    net.add(dict(), coef=2.*rho)
    net.add(pattern_penalty, coef=-2.*rho)


    # Define Hamiltonian
    Jz = 0.5
    H = hamilt.make_H_TFIM(Jz)

    # Define tensors
    d = 2
    D = 32
    np.random.seed(100)
    dtype = float
    AL = ut.random_isometry(d*D, D, dtype).reshape((D,d,D))
    AR = ut.random_isometry(d*D, D, dtype).transpose().reshape((D,d,D))
    C = ut.random_tensor((D,D), dtype)
    C /= np.linalg.norm(C)
    Lambda = ut.random_tensor((D,d,D), dtype)

    # To UniTensors
    H = uniten.ToUniTensor (H, ["s1","s2","s1'","s2'"])
    AL = uniten.ToUniTensor (AL, ["l","s","r"])
    AR = uniten.ToUniTensor (AR, ["l","s","r"])
    C = uniten.ToUniTensor (C, ["l","r"])
    Lambda = uniten.ToUniTensor (Lambda, ["l","s","r"])


    # Define the cost function
    cost_func = cf.CostFunction(net)

    # Put tensors
    cost_func.nets.putTensor("H", H, ["s1","s2","s1'","s2'"])
    cost_func.nets.putTensor("AL", AL, ['l','s',"r"])
    cost_func.nets.putTensor("AR", AR, ['l','s',"r"])
    cost_func.nets.putTensor("C", C, ['l',"r"])
    cost_func.nets.putTensor("AL'", AL.Dagger(), ['l','s',"r"])
    cost_func.nets.putTensor("AR'", AR.Dagger(), ['l','s',"r"])
    cost_func.nets.putTensor("C'", C.Dagger(), ['l',"r"])
    cost_func.nets.putTensor("Lambda", Lambda, ['l','s',"r"])

    exact = -0.317878160486075589

    constraint_crit = 1e-4
    rho_max = 1000
    nGD = 1
    N_linesearch = 20
    for i in range(4001):
        for n in range(nGD):
            cost_func.setGarget("AL")
            AL, value, slope = gd.gradient_descent (cost_func, AL, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["l","s"])

        for n in range(nGD):
            cost_func.setGarget("AR")
            AR, value, slope = gd.gradient_descent (cost_func, AR, step_size=1e-2, N_linesearch=N_linesearch, constraint="UniTen isometry", row_labels=["s","r"])

        for n in range(nGD):
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

        if rho < rho_max:
            rho *= 1.1
        if i % 20 == 0:
            constraint = getDiff(AL, C, AR)
            print(i, cost_func.nets.terms[0], constraint, slope)
            #print(i, cost_func.nets.terms[0].item().real, penalty, slope)

test_gradient_descent_TFIM()
