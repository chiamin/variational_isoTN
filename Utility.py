import numpy as np

def random_tensor (shape, dtype=float):
    if dtype == float:
        A = np.random.random(shape)
    elif dtype == complex:
        R = np.random.random(shape)
        I = np.random.random(shape)
        A = R + 1j*I
    else:
        print("Unsupported type:",dtype)
        raise TypeError
    return A

# W = d x D matrix, d >= D
# W^\dagger W = I
def random_isometry (d, D, dtype=float):
    assert D <= d
    A = random_tensor((d,D), dtype)
    U, S, Vh = np.linalg.svd (A, full_matrices=False)
    assert U.shape == A.shape
    return U

def inner (x1, x2):
    return np.inner(x1.conj().flatten(), x2.flatten())

def dot (A, B):
    assert A.ndim == B.ndim == 2
    return A.conj().T @ B

def outer_dot (A, B):
    assert A.ndim == B.ndim == 2
    return A @ B.conj().T

def is_isometry (W):
    assert W.ndim == 2
    dim = W.shape[1]
    WW = dot(W,W)
    return np.linalg.norm(WW - np.eye(dim)) < 1e-5

