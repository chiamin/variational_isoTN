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

def test_walk_x ():
    d = 2
    D = 20
    np.random.seed(100)
    dtype = float
    ALt = ut.random_isometry(d*D, D, dtype).reshape((D,d,D))
    ARt = ut.random_isometry(d*D, D, dtype).transpose().reshape((D,d,D))
    Ct = ut.random_tensor((D,D), dtype)
    Ct /= np.linalg.norm(Ct)
    Gt = ut.random_tensor((D,d,D), dtype)
    Kt = ut.random_tensor((D,D), dtype)


    AL = uniten.ToUniTensor (ALt, ["l","s","r"])
    AR = uniten.ToUniTensor (ARt, ["l","s","r"])
    C = uniten.ToUniTensor (Ct, ["l","r"])
    G = uniten.ToUniTensor (Gt, ["l","s","r"])
    K = uniten.ToUniTensor (Kt, ["l","r"])

    a = 1.2
    # AL
    res1 = gd.walk_x(ALt, Gt, a, constraint="np AL")
    res2 = gd.walk_x(AL, G, a, constraint="UniTen isometry", row_labels=["l","s"])
    res2 = uniten.ToNpArray(res2)
    print('Is isometry?',ut.is_isometry(res2.reshape((d*D,D))))
    print(np.linalg.norm(res1-res2))

    # AR
    res1 = gd.walk_x(ARt, Gt, a, constraint="np AR")
    res2 = gd.walk_x(AR, G, a, constraint="UniTen isometry", row_labels=["s","r"])
    res2 = uniten.ToNpArray(res2)
    print('Is isometry?',ut.is_isometry(res2.reshape((D,d*D)).T))
    print(np.linalg.norm(res1-res2))

    # C
    res1 = gd.walk_x(Ct, Kt, a, constraint="np normalize")
    res2 = gd.walk_x(C, K, a, constraint="UniTen normalize")
    res2 = uniten.ToNpArray(res2)
    print(np.linalg.norm(res1-res2))

def test_gradient_descent():
    # Define the network pattern
    net = netf.NetworkFunction()
    #     ____________
    #    (_____A______)
    #      |        |
    #     _|________|_
    #    (_____H______)
    #      |        |
    #     _|________|_
    #    (_____A'_____)
    pattern = {
        "A": ["s1","s2"],
        "H": ["s1","s2","s1'","s2'"],
        "A'": ["s1'","s2'"]
    }
    net.add(pattern)

    # Define the cost function
    cost_func = cf.CostFunction(net)
    cost_func.setGarget("A")

    # Define tensors
    H = hamilt.make_H_XX()
    d = 2
    D = 2
    np.random.seed(100)
    A = ut.random_tensor((d,d))

    # Exact solution
    eigenvalues, eigenvectors = np.linalg.eigh(H.reshape((d*d,d*d)))
    print("Exact energy:",eigenvalues)

    H = uniten.ToUniTensor (H, ["s1","s2","s1'","s2'"])
    A = uniten.ToUniTensor (A, ["s1","s2"])
    A /= A.Norm().item()

    # Put tensors
    cost_func.nets.putTensor("H", H, H.labels())
    cost_func.nets.putTensor("A", A, A.labels())
    cost_func.nets.putTensor("A'", A.Dagger(), A.labels())

    for i in range(1001):
        A, value, slope = gd.gradient_descent (cost_func, A, step_size=1e-2, constraint="UniTen normalize")
        if i % 100 == 0:
            print(i, value, slope)

def test_gradient_descent_isometry():
    # Define the network pattern
    net = netf.NetworkFunction()
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

    pattern = pat.Combine (pattern_AA, pattern_AAdag)
    pattern["H"] = ["s1","s2","s1'","s2'"]
    net.add(pattern)

    # Define Hamiltonian
    H = hamilt.make_H_XX()

    # Exact solution
    eigenvalues, eigenvectors = np.linalg.eigh(H.reshape((4,4)))
    print("Exact energy:",eigenvalues)

    # Define tensors
    d = 2
    D = 2
    np.random.seed(100)
    dtype = float
    AL = ut.random_isometry(d, D, dtype).reshape((1,d,D))
    AR = ut.random_isometry(d, D, dtype).transpose().reshape((D,d,1))
    C = ut.random_tensor((D,D), dtype)
    C /= np.linalg.norm(C)

    # To UniTensors
    H = uniten.ToUniTensor (H, ["s1","s2","s1'","s2'"])
    AL = uniten.ToUniTensor (AL, ["l","s","r"])
    AR = uniten.ToUniTensor (AR, ["l","s","r"])
    C = uniten.ToUniTensor (C, ["l","r"])


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

    n = 1
    for i in range(1001):
        cost_func.setGarget("AL")
        AL, value, slope = gd.gradient_descent (cost_func, AL, step_size=1e-2, constraint="UniTen isometry", row_labels=["l","s"])

        cost_func.setGarget("AR")
        AR, value, slope = gd.gradient_descent (cost_func, AR, step_size=1e-2, constraint="UniTen isometry", row_labels=["s","r"])

        cost_func.setGarget("C")
        C, value, slope = gd.gradient_descent (cost_func, C, step_size=1e-2, constraint="UniTen normalize")

        if i % 100 == 0:
            print(i, value, slope)

test_walk_x()
test_gradient_descent()
test_gradient_descent_isometry()
