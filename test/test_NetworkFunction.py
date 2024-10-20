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

def get_energy (H, AL, C, AR):
    #              -2
    #           1   |
    #  -1 ---C------AR--- -3
    C_AR = ncon((C,AR),((-1,1),(1,-2,-3)))
    #    -----AL-----C_AR----
    #    |    |   4    |    |
    #    |  2 |________| 6  |
    #  1 |   (__________)   | 8
    #    |  3 |        | 7  |
    #    |    |   5    |    |
    #    -----AL-----C_AR----
    tmp = ncon((H, AL.conj(), AL, C_AR.conj(), C_AR), ((2,6,3,7),(1,2,4),(1,3,5),(4,6,8),(5,7,8)))
    return tmp

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

def test():
    # Create network for psi
    #
    #         s1              s2
    #         |               |
    #         |   cl     cr   |
    #  l -----AL------C------AR---- r
    pattern_phi = {
        "AL": ["l", "s1", "cl"],
        "C": ["cl", "cr"],
        "AR": ["cr", "s2", "r"],
        "TOUT": ["l", "s1", "s2", "r"]
    }

    # Create conjugate network
    #
    #  l' -----AL'------C'------AR'---- r'
    #          |   cl'     cr'   |
    #          |                 |
    #          s1'               s2'
    pattern_phi_dag = pat.Prime(pattern_phi)
    pattern_phi_dag = pat.ReplaceIndex (pattern_phi_dag, ["l'","r'"], ["l","r"])


    # Create network for energy
    #
    #    -----AL'----C'----AR'---
    #    |    |            |    |
    #    |    |____________|    |
    #    |   (______________)   |
    #    |    |            |    |
    #    |    |            |    |
    #    -----AL-----C-----AR----
    pattern_en = pat.Combine (pattern_phi, pattern_phi_dag)
    pattern_en["H"] = ["s1","s2","s1'","s2'"]


    # Create network for effH_C
    #
    #    -----AL'---    ---AR'---
    #    |    |            |    |
    #    |    |____________|    |
    #    |   (______________)   |
    #    |    |            |    |
    #    |    |            |    |
    #    -----AL-----C-----AR----
    pattern_gradC = copy.copy(pattern_en)
    del pattern_gradC["C'"]


    network = cytnx.Network()
    network.FromString (pat.ToNetworkString(pattern_gradC))


    H = hamilt.make_H_XX()

    d = 2
    D = 20
    np.random.seed(100)
    AL = ut.random_isometry(d*D, D, complex).reshape((D,d,D))
    AR = ut.random_isometry(d*D, D, complex).transpose().reshape((D,d,D))
    C = ut.random_tensor((D,D), complex)
    C /= np.linalg.norm(C)

    # Compute from ncon
    en = get_energy(H, AL, C, AR)
    gradC = gradient1_C (H, AL, C, AR)

    H = uniten.ToUniTensor (H, ["s1","s2","s1'","s2'"])
    AL = uniten.ToUniTensor (AL, ["l","s","r"])
    AR = uniten.ToUniTensor (AR, ["l","s","r"])
    C = uniten.ToUniTensor (C, ["l","r"])


    # Compute effH_C from network
    network.PutUniTensor("H", H, ['s1','s2',"s1'","s2'"])
    network.PutUniTensor("AL", AL, ['l','s',"r"])
    network.PutUniTensor("AR", AR, ['l','s',"r"])
    network.PutUniTensor("AL'", AL.Dagger(), ['l','s',"r"])
    network.PutUniTensor("AR'", AR.Dagger(), ['l','s',"r"])
    network.PutUniTensor("C", C, ['l',"r"])

    T = network.Launch()
    T = T.get_block().numpy()
    print(np.linalg.norm(T - gradC))

    # Define a NetworkFunction for energy
    gfunc = netf.NetworkFunction()
    gfunc.add(pattern_en)
    # Put tensors
    gfunc.putTensor("H", H, ['s1','s2',"s1'","s2'"])
    gfunc.putTensor("AL", AL, ['l','s',"r"])
    gfunc.putTensor("AR", AR, ['l','s',"r"])
    gfunc.putTensor("C", C, ['l',"r"])
    gfunc.putTensor("AL'", AL.Dagger(), ['l','s',"r"])
    gfunc.putTensor("AR'", AR.Dagger(), ['l','s',"r"])
    gfunc.putTensor("C'", C.Dagger(), ['l',"r"])
    # NetworkFunction for gradient
    T = gfunc.gradient("C")
    T = T.get_block().numpy()
    print(np.linalg.norm(0.5*T - gradC))


a = cytnx.UniTensor.arange(2*3*4).reshape(2,3,4).relabels(["a","b","c"])
b = cytnx.UniTensor.arange(2*3*4).reshape(4,3,2).relabels(["a","b","c"]).permute(["c","b","a"])
c = a+b
print(c.labels())
a = cytnx.UniTensor.arange(2*3*4).reshape(2,3,4).relabels(["a","b","c"])
b = cytnx.UniTensor.arange(2*3*4).reshape(2,3,4).relabels(["a'","b'","c"])
c = cytnx.Contract(a, b)
print(c.labels())
exit()


test()

