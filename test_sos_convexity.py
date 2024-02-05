import sys
import importlib
import cvxpy as cp 

import numpy as np
import matplotlib.pyplot as plt

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

objective = cp.Minimize( 1.0 )
constraints = [ P >> 0, SOSConvexMatrix >> 0 ]

c = [0.0, -5.0]
m_c = kernel.function(c)
constraints.append( m_c.T @ P @ m_c == 0.0 )

level_set = 10
p = [0.0, 1.0]
m_p = kernel.function(p)
constraints.append( m_p.T @ P @ m_p == 2.0*level_set )

problem = cp.Problem(objective, constraints)
problem.solve(verbose=True)

print(f"Eigenvalues of P = {np.linalg.eigvals(P.value)}")
print(f"Eigenvalues of Hessian form = {np.linalg.eigvals(SOSConvexMatrix.value)}")