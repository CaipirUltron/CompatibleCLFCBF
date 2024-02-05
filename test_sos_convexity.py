import cvxpy as cp 

import itertools
import numpy as np
import matplotlib.pyplot as plt

from common import create_quadratic, rot2D
from functions import Kernel
from controllers.equilibrium_algorithms import compute_equilibria, plot_invariant, closest_to_image

# ---------------------------------------------- Define kernel function ----------------------------------------------------
initial_state = [0.5, 6.0]
kernel = Kernel(*initial_state, degree=2)
kernel_dim = kernel.kernel_dim
print(kernel)

A_list = kernel.get_A_matrices()
n = len(A_list)
p = kernel.kernel_dim

P = cp.Variable( (p,p), symmetric=True )
SOSConvexMatrix = cp.bmat([[ Aj.T @ ( Ai.T @ P + P @ Ai ) + ( Ai.T @ P + P @ Ai ) @ Aj for Aj in A_list ] for Ai in A_list ])

objective = cp.Minimize( 0.0 )

c = [0.0, -5.0]
m_c = kernel.function(c)
Pquad = create_quadratic(eigen=[0.1, 0.1], R=rot2D(0.0), center=c, kernel_dim=kernel_dim)
constraints = [ P >> Pquad, SOSConvexMatrix >> 0 ]
constraints.append( m_c.T @ P @ m_c == 0.0 )

pivot = [0.0, 3.0]
m_p = kernel.function(pivot)
Jm_p = kernel.jacobian(pivot)

gradient = [ 0.0, 1.0 ]
gradient_norm = cp.Variable()
normalized = gradient/np.linalg.norm(gradient)

constraints.append( Jm_p.T @ P @ m_p == gradient_norm * normalized )
constraints.append( gradient_norm >= 0 )

problem = cp.Problem(objective, constraints)
problem.solve(verbose=True)

print(f"Vcenter = {0.5 * m_c.T @ P.value @ m_c}")

print(f"Eigenvalues of P = {np.linalg.eigvals(P.value)}")
# print(f"Eigenvalues of Hessian form = {np.linalg.eigvals(SOSConvexMatrix.value)}")

# P = np.random.rand(p,p)
# P = P.T @ P
# for Ai in A_list:
#     for Aj in A_list:
#         M = Aj.T @ ( Ai.T @ P + P @ Ai ) + ( Ai.T @ P + P @ Ai ) @ Aj
        
#         for line in M:
#             l = []
#             for ele in line:
#                 l.append('{:3.1f}'.format(ele))
#             print(l)
#         print("\n")