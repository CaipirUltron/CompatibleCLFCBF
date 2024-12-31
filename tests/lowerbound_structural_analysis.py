'''
This script uses symbolic computation to explore the internal structure of the symmetric linear operators
L(P) and R(P) >> 0, components of the matrix L(P) + R(P) for the lowerbound on the maximum eigenvalue of the
Hessian matrix associated to the polynomial kernel function f(x) = ½ k(x).T P k(x) + Constant:

                        n λ_max( ∇ x ∇ f ) >= k(x).T ( L(P) + R(P) ) k(x)

where n is the state space dimension, k(x) is the polynomial kernel map of max. degree d, and ∇ x ∇ f is the 
Hessian matrix of the kernel function.

L(P) and R(P) can be written as the following block matrices, valid for all state dimensions n and max. degree d:
       | L_11  L_12 L_13 |          | R_11  R_12  0 |
L(P) = | L_12'  0    0   |   R(P) = | R_12' R_22  0 | >> 0 ,
       | L_13'  0    0   |,         |  0     0    0 |
where each block L_ij, R_ij has the same size and its own particular dependencies on specific terms of P.

This script computes: (i) the symbolic expressions for the blocks L_ij, R_ij
                      (ii) the block sizes
                      (iii) the dependencies of each blocks on the terms of P

PRATICAL LIMITATION: 
n = 3, d = 6 causes a stack overflow in the python interpreter.
This script was used to confirm the hypothesis that the block sizes sequences follow the sides of Pascal's triangle, 
as a function of the state dimension n and the max. polynomial degree d.
'''

import sympy as sym

from contextlib import redirect_stdout
from functions import Kernel
from common import *

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=2, degree=4)
kernel_dim = kernel._num_monomials
# --------------------------------------------------------------------------------------------------------------------------

As = kernel.Asum
As2 = As @ As

P = sym.MatrixSymbol('P', kernel_dim, kernel_dim).as_explicit()
Psym = sym.Matrix(kernel_dim, kernel_dim, lambda i, j: P[min(i,j),max(i,j)])

Lsym = lyap( As2.T, Psym )
Rsym = 2 * As.T @ Psym @ As
Msym = lyap( As.T, lyap( As.T, Psym ) )

Psymbols = Psym.atoms(sym.matrices.expressions.matexpr.MatrixElement)
R_dependencies, R_structure, P_structure, blk_sizes = kernel.show_structure(Rsym)
L_dependencies, L_structure, P_structure, blk_sizes = kernel.show_structure(Lsym)
M_dependencies, M_structure, P_structure, blk_sizes = kernel.show_structure(Msym)

def printing():
    ''' Prints '''
    print(kernel)

    print("\nP symbols = ",end='')
    sym.pprint(Psymbols, wrap_line=False)

    print(f"\nBlock sizes = {blk_sizes}")

    print(f"\nP structure = ",end='')
    for i, line in enumerate(P_structure):
        if i > 0: print(f"              ",end='')
        sym.pprint(line, wrap_line=False)

    print(f"\nR(P) >> 0:")

    print(f"\nR(P) dependencies = ",end='')
    for i, line in enumerate(R_dependencies):
        if i > 0: print(f"                    ",end='')
        sym.pprint(line, wrap_line=False)

    print(f"\nR(P) structure = ",end='')
    for i, line in enumerate(R_structure):
        if i > 0: print(f"                 ",end='')
        sym.pprint(line, wrap_line=False)

    print(f"\nL(P): ")

    print(f"\nL(P) dependencies = ",end='')
    for i, line in enumerate(L_dependencies):
        if i > 0: print(f"                    ",end='')
        sym.pprint(line, wrap_line=False)

    print(f"\nL(P) structure = ",end='')
    for i, line in enumerate(L_structure):
        if i > 0: print(f"                 ",end='')
        sym.pprint(line, wrap_line=False)

    print(f"\nL(P) + R(P):")

    print(f"\nL(P) + R(P) dependencies = ",end='')
    for i, line in enumerate(M_dependencies):
        if i > 0: print(f"                           ",end='')
        sym.pprint(line, wrap_line=False)

    print(f"\nL(P) + R(P) structure = ",end='')
    for i, line in enumerate(M_structure):
        if i > 0: print(f"                        ",end='')
        sym.pprint(line, wrap_line=False)

# redirect = True
redirect = False
if redirect:
    with open(f'logs/lowerbound_structure_n={kernel._dim}_d={kernel._degree}.txt', 'w') as f:
        with redirect_stdout(f):
            printing()
else:
    printing()

sym.pprint(Msym)