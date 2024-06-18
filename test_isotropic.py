import numpy as np
import scipy as sp
from functions import Kernel

np.set_printoptions(precision=2, suppress=True)

n, d = 2, 2
kernel = Kernel(dim=n, degree=d)
print(kernel)
p = kernel._num_monomials
A_list = kernel.get_A_matrices()

slk_gain, ctrl_gain = 1, 1

P = np.random.randn(p,p)
P = P.T @ P

Q = np.random.randn(p,p)
Q = Q.T @ Q

S = np.random.randn(p,p)

l = np.random.rand()
l = 0.0

List = [ Ai.T @ ( S + l*Q - slk_gain*ctrl_gain*P ) for Ai in A_list ]
nulls = [ sp.linalg.null_space(Li) for Li in List ]
ranks = [ np.linalg.matrix_rank(Li) for Li in List ]

for k, L in enumerate(List): 
    print(f"L{k+1} = \n{L}")
    print(f"nullspace(L{k+1}) = \n{nulls[k]}")
    print(f"rank(L{k+1}) = {ranks[k]}")
