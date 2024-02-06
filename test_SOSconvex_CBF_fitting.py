import cvxpy as cp 

import itertools
import numpy as np
import matplotlib.pyplot as plt

from common import create_quadratic, rot2D, box
from functions import Kernel, KernelBarrier
from controllers.equilibrium_algorithms import compute_equilibria, plot_invariant, closest_to_image

# ---------------------------------------------- Define kernel function ----------------------------------------------------
initial_state = [0.5, 6.0]
kernel = Kernel(*initial_state, degree=2)
kernel_dim = kernel.kernel_dim
print(kernel)

A_list = kernel.get_A_matrices()
n = len(A_list)
p = kernel.kernel_dim

# ---------------------------------------------- Define basic SOS constraints ---------------------------------------------
P = cp.Variable( (p,p), symmetric=True )
SOSConvexMatrix = cp.bmat([[ Aj.T @ ( Ai.T @ P + P @ Ai ) + ( Ai.T @ P + P @ Ai ) @ Aj for Aj in A_list ] for Ai in A_list ])
constraints = [ 
                P >> 0, 
                # SOSConvexMatrix >> 0
                ]

# Define CBF center
c = [0.0, 4.0]
m_c = kernel.function(c)
constraints.append( m_c.T @ P @ m_c == 0.0 )

# Creates box points to be fitted
pts = box( center=c, height=5, width=5, angle=15, res=3 )

# ----------------------------------------------- Plotting ----------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Invariant set plot for Kernel-based CLF-CBFs")
ax.set_aspect('equal', adjustable='box')

for pt in pts:
    ax.plot(pt[0], pt[1], 'k*')

cbf = KernelBarrier(*initial_state, kernel=kernel, boundary=pts, centers=[c])
cbf.is_sos_convex(verbose=True)

print(f"h at center = {cbf.function(c)}")

limits = (9*np.array([[-1, 1],[-1, 1]])).tolist()

ax.set_xlim(limits[0][0], limits[0][1])
ax.set_ylim(limits[1][0], limits[1][1])

num_lvls = 10
contour_boundary = cbf.plot_levels(levels = [ 0.5*(k - (num_lvls - 1))/(num_lvls - 1) for k in range(0,num_lvls,1) ], ax=ax, limits=limits)
# contour_boundary = cbf.plot_levels(levels = [ 2*k for k in range(0,num_lvls,1) ], ax=ax, limits=limits)

# A1, A2 = A_list[0], A_list[1]
# P1 = ( A1.T @ P.value + P.value @ A1 )
# P2 = ( A2.T @ P.value + P.value @ A2 )

# Aj.T @ Pi + Pi @ Aj
# P11 = A1.T @ P1 + P1 @ A1
# P12 = A2.T @ P1 + P1 @ A2
# P21 = A1.T @ P2 + P2 @ A1
# P22 = A2.T @ P2 + P2 @ A2

np.set_printoptions(precision=2, suppress=False)

cbf.SOS_convexity()

plt.show()