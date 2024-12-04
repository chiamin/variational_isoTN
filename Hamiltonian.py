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

def make_H_TFIM (Jz, hx):
    sx = 0.5*np.array([[0,1],[1,0]],dtype=float)
    sz = 0.5*np.array([[1,0],[0,-1]],dtype=float)
    I = np.eye(2)
    def prod (A, B):
        return ncon ([A,B], ((-1,-3), (-2,-4)))
    H = hx * prod (sx, sx)
    H += 0.5*Jz * prod(sz, I)
    H += 0.5*Jz * prod(I, sz)
    return H


def make_H_TFIM2 (Jx, hz):
    sx = 0.5*np.array([[0,1],[1,0]],dtype=float)
    sz = 0.5*np.array([[1,0],[0,-1]],dtype=float)
    I = np.eye(2)
    def prod (A, B):
        return ncon ([A,B], ((-1,-3), (-2,-4)))
    H = Jx * prod (sx, sx)
    H += (1/3)*hz * prod(sz, I)
    H += (1/3)*hz * prod(I, sz)
    return H

def make_Sz():
    return np.array([[1,0],[0,-1]],dtype=float)

def make_Hx_TFIM2 (Jx, hz):
    sx = 0.5*np.array([[0,1],[1,0]],dtype=float)
    sz = 0.5*np.array([[1,0],[0,-1]],dtype=float)
    I = np.eye(2)
    def prod (A, B):
        return ncon ([A,B], ((-1,-3), (-2,-4)))
    H = Jx * prod (sx, sx)
    H += hz * prod(sz, I)
    return H

def make_Hy_TFIM2 (Jx, hz):
    sx = 0.5*np.array([[0,1],[1,0]],dtype=float)
    def prod (A, B):
        return ncon ([A,B], ((-1,-3), (-2,-4)))
    H = Jx * prod (sx, sx)
    return H


def make_H_Heisenberg (J):
    sx = 0.5*np.array([[0,1],[1,0]],dtype=float)
    sz = 0.5*np.array([[1,0],[0,-1]],dtype=float)
    sp = np.array([[0,1],[0,0]],dtype=float)
    sm = np.array([[0,0],[1,0]],dtype=float)
    I = np.eye(2)
    def prod (A, B):
        return ncon ([A,B], ((-1,-3), (-2,-4)))
    H = J * prod (sz, sz)
    H += 0.5*J * prod (sp, sm)
    H += 0.5*J * prod (sm, sp)
    return H
