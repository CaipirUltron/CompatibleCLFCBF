import numpy as np
import sympy as sym

import cvxpy as cp
import matplotlib.pyplot as plt

from functions import Kernel
from common import *

# ---------------------------------------------- Define kernel function ----------------------------------------------------
d=3
kernel = Kernel(dim=2, degree=d)
kernel_dim = kernel._num_monomials

As = kernel.Asum
As2 = As @ As
Ip = np.eye(kernel_dim)
kron = np.kron(As2.T, Ip) + np.kron(Ip, As2.T)

P = sym.MatrixSymbol('P', kernel_dim, kernel_dim).as_explicit()
Psym = sym.Matrix(kernel_dim, kernel_dim, lambda i, j: P[min(i,j),max(i,j)])

Lsym = lyap( As2.T, Psym )

print(f"P = ")
sym.pprint(Psym)

print(f"\n L = ")
sym.pprint(Lsym)

P_null_sym = sym.zeros( kernel_dim, kernel_dim )
P_filtered_sym = sym.zeros( kernel_dim, kernel_dim )
for (i,j) in itertools.product(range(kernel_dim), range(kernel_dim)):

    if Lsym.has(Psym[i,j]):
        P_filtered_sym[i,j] = Psym[i,j]
    else:
        P_null_sym[i,j] = Psym[i,j]

print(f"\n P filtered = ")
sym.pprint(P_filtered_sym)

print(f"\n P null = ")
sym.pprint(P_null_sym)
