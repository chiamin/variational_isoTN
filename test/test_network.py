import os, sys, copy
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0,'/home/chiamin/Cytnx/Install/')
import cytnx
import numpy as np
sys.path.append('../')
import Pattern as pat
from ncon import ncon

# W = d x D matrix, d >= D
# W^\dagger W = I
def random_isometry (d, D):
    assert D <= d
    A = np.random.random((d,D))
    U, S, Vh = np.linalg.svd (A, full_matrices=False)
    assert U.shape == A.shape
    return U

def make_H_XX ():
    sx = 0.5*np.array([[0,1],[1,0]],dtype=complex)
    sy = 0.5*np.array([[0,-1j],[1j,0]],dtype=complex)
    sz = 0.5*np.array([[1,0],[0,-1]],dtype=float)
    sp = np.array([[0,1],[0,0]],dtype=float)
    sm = np.array([[0,0],[1,0]],dtype=float)
    I = np.eye(2)
    def prod (A, B):
        return ncon ([A,B], ((-1,-3), (-2,-4)))
    H = 0.5 * prod(sp, sm)
    H += 0.5 * prod(sm, sp)
    return H.real

def ToUniTensor (NpArray, labels):
    assert NpArray.ndim == len(labels)
    A = cytnx.UniTensor (cytnx.from_numpy(NpArray))
    A.set_labels(labels)
    return A

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

def get_effH_C (H, AL, AR):
    #    -----AL--- -1   -2 ---AR----
    #    |    |                |    |
    #    |  2 |________________| 4  |
    #  1 |   (__________________)   | 6
    #    |  3 |                | 5  |
    #    |    |                |    |
    #    -----AL--- -3   -4 ---AR----
    tmp = ncon((H, AL.conj(), AL, AR.conj(), AR), ((2,4,3,5),(1,2,-1),(1,3,-3),(-2,4,6),(-4,5,6)))
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
    #    -----AL'--     ---AR'---
    #    |    |            |    |
    #    |    |____________|    |
    #    |   (______________)   |
    #    |    |            |    |
    #    |    |            |    |
    #    -----AL---     ---AR----
    pattern_effHC = copy.copy(pattern_en)
    del pattern_effHC["C"]
    del pattern_effHC["C'"]


    network_en = cytnx.Network()
    network_en.FromString (pat.ToNetworkString(pattern_en))

    network_effHC = cytnx.Network()
    network_effHC.FromString (pat.ToNetworkString(pattern_effHC))


    H = make_H_XX()

    d = 2
    D = 20
    np.random.seed(100)
    AL = random_isometry (d*D, D).reshape((D,d,D))
    AR = random_isometry (d*D, D).transpose().reshape((D,d,D))
    C = np.random.random((D,D))
    C /= np.linalg.norm(C)

    # Compute from ncon
    en = get_energy(H, AL, C, AR)
    effH_C = get_effH_C (H, AL, AR)



    aa = ToUniTensor (AL, ["l","s","r"])
    bb = ToUniTensor (AR, ["a","b","c"])

    H = ToUniTensor (H, ["s1","s2","s1'","s2'"])
    AL = ToUniTensor (AL, ["l","s","r"])
    AR = ToUniTensor (AR, ["l","s","r"])
    C = ToUniTensor (C, ["l","r"])


    # Compute energy from network
    network_en.PutUniTensor("H", H, ['s1','s2',"s1'","s2'"])
    network_en.PutUniTensor("AL", AL, ['l','s',"r"])
    network_en.PutUniTensor("AR", AR, ['l','s',"r"])
    network_en.PutUniTensor("C", C, ['l',"r"])
    network_en.PutUniTensor("AL'", AL.Dagger(), ['l','s',"r"])
    network_en.PutUniTensor("AR'", AR.Dagger(), ['l','s',"r"])
    network_en.PutUniTensor("C'", C.Dagger(), ['l',"r"])

    E = network_en.Launch()
    print(E.item(), en)


    # Compute effH_C from network
    network_effHC.PutUniTensor("H", H, ['s1','s2',"s1'","s2'"])
    network_effHC.PutUniTensor("AL", AL, ['l','s',"r"])
    network_effHC.PutUniTensor("AR", AR, ['l','s',"r"])
    network_effHC.PutUniTensor("AL'", AL.Dagger(), ['l','s',"r"])
    network_effHC.PutUniTensor("AR'", AR.Dagger(), ['l','s',"r"])

    T = network_effHC.Launch()
    print(T.labels())
    print(network_effHC)
    T = T.get_block().numpy()
    print(np.linalg.norm(T - effH_C))

test()

