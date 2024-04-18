import numpy as np
import sympy as sym

import cvxpy as cp
import matplotlib.pyplot as plt

from functions import Kernel
from common import *

# ---------------------------------------------- Define kernel function ----------------------------------------------------
d=2
kernel = Kernel(dim=2, degree=d)
kernel_dim = kernel._num_monomials
print(kernel)

As = kernel.Asum
As2 = As @ As
Ip = np.eye(kernel_dim)
kron = np.kron(As2.T, Ip) + np.kron(Ip, As2.T)

P = sym.MatrixSymbol('P', kernel_dim, kernel_dim).as_explicit()
Psym = sym.Matrix(kernel_dim, kernel_dim, lambda i, j: P[min(i,j),max(i,j)])

Lsym = lyap( As2.T, Psym )
Rsym = 2 * As.T @ Psym @ As
Msym = lyap( As.T, lyap( As.T, Psym ) )

print(f"\n R = ")
sym.pprint(Rsym)

print("\nR structure = \n")
R_dependencies, P_structure = kernel.show_structure(Rsym)
for line in R_dependencies:
    print(line)

print(f"\nP structure = ")
for line in P_structure:
    sym.pprint(line)

print(f"\n L = ")
sym.pprint(Lsym)

print("\nL structure = \n")
L_dependencies, P_structure = kernel.show_structure(Lsym)
for line in L_dependencies:
    print(line)

print(f"\nP structure = ")
for line in P_structure:
    sym.pprint(line)

print(f"\n M = ")
sym.pprint(Msym)

print("\nM structure = \n")
M_dependencies, P_structure = kernel.show_structure(Msym)
for line in M_dependencies:
    print(line)

print(f"\nP structure = ")
for line in P_structure:
    sym.pprint(line)

# P_null_sym = sym.zeros( kernel_dim, kernel_dim )
# P_filtered_sym = sym.zeros( kernel_dim, kernel_dim )
# for (i,j) in itertools.product(range(kernel_dim), range(kernel_dim)):
#     if Lsym.has(Psym[i,j]):
#         P_filtered_sym[i,j] = Psym[i,j]
#     else:
#         P_null_sym[i,j] = Psym[i,j]

# print(f"\n P filtered = ")
# sym.pprint(P_filtered_sym)

# print(f"\n P null = ")
# sym.pprint(P_null_sym)

# n = 3

# Peffec = Psym
# for (i,j) in itertools.product(range(kernel_dim), range(kernel_dim)):
#     if i > n-1 and j <= n-1: Peffec[i,j] = 0
#     if j > n-1 and i <= n-1: Peffec[i,j] = 0

# print(f"\n Peffec = ")
# sym.pprint(Peffec)

# Leffec = lyap( As2.T, Peffec)
# print(f"\n Leffec = ")
# sym.pprint(Leffec)