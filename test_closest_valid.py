import numpy as np
import sympy as sym
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt

from functions import Kernel, KernelQuadratic, LeadingShape, KernelLyapunov, KernelBarrier
from common import create_quadratic, rot2D, box, polygon, lyap

limits = 12*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test level sets")
ax.set_aspect('equal', adjustable='box')

xmin, xmax, ymin, ymax = limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=2, degree=3)
kernel_dim = kernel._num_monomials

epsilon = 1e-0
As = kernel.Asum
I = np.eye(kernel_dim)
kron_matrix = ( np.kron(I,As.T) + np.kron(As, I) )
kron_matrix = kron_matrix @ kron_matrix

# print(f"Eigenvals of kronecker matrix:\n {np.linalg.eigvals(kron_matrix)}")

Psym = kernel._sym_shape_matrix
# Qsym = lyap(As.T, Psym)
# Rsym = lyap(As.T, Qsym)

P_var = cp.Variable( (kernel_dim, kernel_dim), symmetric=True )
R_var = cp.Variable( (kernel_dim, kernel_dim), symmetric=True )
Pnom_var = cp.Parameter( (kernel_dim, kernel_dim), symmetric=True )

n = kron_matrix @ np.random.rand(kernel_dim**2)

print(f"zeros = {np.where(n==0)}")

# print(f"Lower = {kernel._lowerbound_matrix2(P_var)}")

max_lambdaP = 100
cost = cp.norm(P_var - Pnom_var) + cp.norm( kron_matrix @ cp.vec(P_var) - cp.vec(R_var) )
constraints = [ P_var >> 0 , R_var >> 0 ]
problem = cp.Problem( cp.Minimize(cost), constraints )

def find_closest_valid(Pnom: np.ndarray, verbose=False) -> np.ndarray:
    ''' Find closest shape matrix to Pnom belonging to family of valid CLFs (without local minima) '''
    Pnom_var.value = Pnom
    problem.solve(verbose=verbose)
    return P_var.value

# P = np.random.randn(kernel_dim, kernel_dim)
# P = P.T @ P
# Pclosest = find_closest_valid(P, verbose=True)

# print(f"Eigenvals of P:\n {np.linalg.eigvals(P)}")
# print(f"Eigenvals of lowerbound:\n {np.linalg.eigvals(kernel._lowerbound_matrix2(Pclosest))}")