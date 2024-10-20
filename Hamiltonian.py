import numpy as np
from ncon import ncon

def make_H_XX ():
    sx = 0.5*np.array([[0,1],[1,0]],dtype=complex)
    sp = np.array([[0,1],[0,0]],dtype=float)
    sm = np.array([[0,0],[1,0]],dtype=float)
    I = np.eye(2)
    def prod (A, B):
        return ncon ([A,B], ((-1,-3), (-2,-4)))
    H = 0.5 * prod(sp, sm)
    H += 0.5 * prod(sm, sp)
    return H

def make_H_TFIM (Jz):
    sx = 0.5*np.array([[0,1],[1,0]],dtype=float)
    sz = 0.5*np.array([[1,0],[0,-1]],dtype=float)
    I = np.eye(2)
    def prod (A, B):
        return ncon ([A,B], ((-1,-3), (-2,-4)))
    H = prod (sx, sx)
    H += 0.5*Jz * prod(sz, I)
    H += 0.5*Jz * prod(I, sz)
    return H
