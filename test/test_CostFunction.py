import copy, sys
sys.path.insert(0,'/home/chiamin/Cytnx/Install/')
import cytnx
import numpy as np
sys.path.append('../')
import Pattern as pat
import NetworkFunction as netf
import Hamiltonian as hamilt
import UniTensorTools as uniten
import Utility as ut
from ncon import ncon

def CostFunctionPattern ():
    cfun = netf.NetworkFunction()

    # 1. Energy term
    #
    # Create network for psi
    #
    #         s1              s2
    #         |               |
    #         |   cl     cr   |
    #  l -----AL------C------AR---- r
    pattern_AA = {
        "AL": ["l", "s1", "cl"],
        "C": ["cl", "cr"],
        "AR": ["cr", "s2", "r"],
    }

    # Create conjugate network
    #
    #   l -----AL'------C'------AR'---- r
    #          |   cl'     cr'   |
    #          |                 |
    #          s1'               s2'
    pattern_AAdag = pat.Prime(pattern_AA)
    pattern_AAdag = pat.ReplaceIndex (pattern_AAdag, ["l'","r'"], ["l","r"])


    # Create network for energy
    #
    #    -----AL'----C'----AR'---
    #    |    |            |    |
    #    |    |____________|    |
    #    |   (______________)   |
    #    |    |            |    |
    #    |    |            |    |
    #    -----AL-----C-----AR----
    pattern_en = pat.Combine (pattern_AA, pattern_AAdag)
    pattern_en["H"] = ["s1","s2","s1'","s2'"]



    # 2. Constraint penalty
    #    -----C'--------AR'---------
    #    |         cr   |          |
    #  l |              | s        | r
    #    |              |          |
    #    ---------------AL------C---
    #                       cl
    pattern_P = {
        "AL": ["l", "s", "cl"],
        "C": ["cl", "r"],
        "AR'": ["cr", "s", "r"],
        "C'": ["l","cr"]
    }


    # Update to cost function as the Lagrainge
    cfun.add(pattern_en)
    cfun.add(pattern_P)

    return cfun


def gradient1_C (H, AL, C, AR):
    #       -2
    #        |    1
    #  -1 ---AL------C--- -3
    AL_C = ncon((AL,C),((-1,-2,1),(1,-3)))
    #     ------AL--- -1
    #     |      |         -2
    #     |    1 |__________|
    #   2 |     (______H_____)
    #     |    3 |          |
    #     |      |         -4
    #     ---(AL_C)--- -3
    L = ncon((AL.conj(), H, AL_C), ((2,1,-1),(1,-2,3,-4),(2,3,-3)))
    #   --------o--- -1   -2 ---AR-------
    #   |       |               |       |
    #   |       |_______________| 4     |
    #   |      (_________________)      | 3
    #   |       |               | 2     |
    #   |       |           1   |       |
    #   --------o---------------AR-------
    tmp = ncon((L,AR.conj(), AR),((-1,4,1,2),(-2,4,3),(1,2,3)))
    return tmp

def gradient2_C (AL, AR, C):
    #   --- -1   -2 ---AR----------
    #   |              |          |
    #   |              | 3        | 2
    #   |              |          |
    #   ---------------AL------C---
    #                       1
    tmp = ncon((AR.conj(),AL,C), ((-2,3,2),(-1,3,1),(1,2)))
    return tmp

def test():
    # Define tensors
    H = hamilt.make_H_XX()
    d = 2
    D = 20
    np.random.seed(100)
    dtype = complex
    AL = ut.random_isometry(d*D, D, dtype).reshape((D,d,D))
    AR = ut.random_isometry(d*D, D, dtype).transpose().reshape((D,d,D))
    C = ut.random_tensor((D,D), dtype)
    C /= np.linalg.norm(C)

    # Compute the gradient using ncon
    g = 2*gradient1_C(H, AL, C, AR) + 2*gradient2_C(AL, AR, C)

    # To UniTensors
    H = uniten.ToUniTensor (H, ["s1","s2","s1'","s2'"])
    AL = uniten.ToUniTensor (AL, ["l","s","r"])
    AR = uniten.ToUniTensor (AR, ["l","s","r"])
    C = uniten.ToUniTensor (C, ["l","r"])

    # Compute the gradient using NetworkFunction
    func = CostFunctionPattern()
    func.putTensor("H", H, ['s1','s2',"s1'","s2'"])
    func.putTensor("AL", AL, ['l','s',"r"])
    func.putTensor("AR", AR, ['l','s',"r"])
    func.putTensor("C", C, ['l',"r"])
    func.putTensor("AL'", AL.Dagger(), ['l','s',"r"])
    func.putTensor("AR'", AR.Dagger(), ['l','s',"r"])
    func.putTensor("C'", C.Dagger(), ['l',"r"])
    gradC = func.gradient("C")
    gg = uniten.ToNpArray(gradC)

    print(np.linalg.norm(g - gg))


test()
