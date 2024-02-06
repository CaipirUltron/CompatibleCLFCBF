import cvxpy as cp 

import itertools
import numpy as np
import matplotlib.pyplot as plt

from common import create_quadratic, rot2D
from functions import Kernel, KernelLyapunov
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

c = [0.0, -5.0]
m_c = kernel.function(c)
# Pquad = create_quadratic(eigen=[0.1, 0.1], R=rot2D(0.0), center=c, kernel_dim=kernel_dim)
constraints = [ P >> 0.0, SOSConvexMatrix >> 0 ]
constraints.append( m_c.T @ P @ m_c == 0.0 )

V = 10
num_pts = 4
for _ in range(num_pts):
    p = np.random.rand(n)
    m_p = kernel.function(p)
    constraints.append( 0.5 * m_p.T @ P @ m_p == V )

# gradient = [ 0.0, 1.0 ]
# gradient_norm = cp.Variable()
# normalized = gradient/np.linalg.norm(gradient)

# constraints.append( Jm_p.T @ P @ m_p == gradient_norm * normalized )
# constraints.append( gradient_norm >= 0 )

problem = cp.Problem(objective, constraints)
problem.solve(verbose=True)

clf = KernelLyapunov(*initial_state, kernel=kernel, P = P.value)

print(f"Vcenter = {0.5 * m_c.T @ P.value @ m_c}")
print(f"Eigenvalues of P = {np.linalg.eigvals(P.value)}")
# print(f"Eigenvalues of Hessian form = {np.linalg.eigvals(SOSConvexMatrix.value)}")

# ----------------------------------------------- Plotting ----------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Invariant set plot for Kernel-based CLF-CBFs")
ax.set_aspect('equal', adjustable='box')

limits = (9*np.array([[-1, 1],[-1, 1]])).tolist()

ax.set_xlim(limits[0][0], limits[0][1])
ax.set_ylim(limits[1][0], limits[1][1])

num_lvls = 10
contour_boundary = clf.plot_levels(levels = [ 10*k for k in range(0,num_lvls,1) ], ax=ax, limits=limits)
plt.show()