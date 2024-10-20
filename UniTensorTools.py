import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0,'/home/chiamin/Cytnx/Install/')
import cytnx

def ToUniTensor (NpArray, labels=[]):
    assert NpArray.ndim == len(labels)
    A = cytnx.UniTensor (cytnx.from_numpy(NpArray))
    if len(labels) != 0:
        A.set_labels(labels)
    return A

def ToNpArray (UniTensor):
    if UniTensor.is_blockform():
        A = cytnx.UniTensor.zeros(UniTensor.shape())
        A.convert_from(UniTensor)
        A = A.get_block().numpy()
    else:
        A = UniTensor.get_block().numpy()
    return A

def ToReal (T):
    res = cytnx.UniTensor(T.bonds(), T.labels(), dtype=cytnx.Type.Double)
    B = T.get_block()
    res.put_block(B.real())
    return res
