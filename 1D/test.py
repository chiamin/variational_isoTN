import os, sys, copy
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0,'/home/chiamin/Cytnx/Install/')
import cytnx
import numpy as np



string = ["AL1: l1, 18, 16, 6",
"AR1: 7, 2, 12, 10",
"AL2: 19, 4, 14, 6",
"AR2: 1, 8, 11, 10",
"C1: 16, 7, 15",
"C2: 14, 1, 15",
"Hx: 18, 2, s11', 9",
"AR1': 17, 21, 12, 20",
"AL2': 19, 4, 3, c1'",
"AR2': 13, 8, 11, 20",
"C1': d1', 17, 5",
"C2': 3, 13, 5",
"TOUT: l1, s11', d1', c1'"]


network = cytnx.Network()
network.FromString (string)

AL1 ['l1', 's11', 'd1', 'c1']
AR1 ['d2', 's12', 'r1', 'c2']
AL2 ['l2', 's21', 'u1', 'c1']
AR2 ['u2', 's22', 'r2', 'c2']
C1 ['d1', 'd2', 'c']
C2 ['u1', 'u2', 'c']
Hx ['s11', 's12', "s11'", "s22'"]
AR1' ["d2'", "s12'", 'r1', "c2'"]
AL2' ['l2', 's21', "u1'", "c1'"]
AR2' ["u2'", 's22', 'r2', "c2'"]
C1' ["d1'", "d2'", "c'"]
C2' ["u1'", "u2'", "c'"]
TOUT ['l1', "s11'", "d1'", "c1'"]
