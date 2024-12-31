import cvxpy as cp 

import itertools
import numpy as np
import matplotlib.pyplot as plt

from common import create_quadratic, rot2D, box, polygon
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
constraints = [ P >> 0, 
                # SOSConvexMatrix >> 0
                ]

# Define CBF centers
centers = [ [ 0.0, 0.0 ], [ -4.0, 3.0 ], [ 4.0, 3.0 ] ]

# Fits CBF to a box-shaped obstacle
pts = [ [ 5.0,-1.0 ], [ 5.0, 3.0 ], [ 3.0, 3.0 ], [ 3.0, 1.0 ],
        [-3.0, 1.0 ], [-3.0, 3.0 ], [-5.0, 3.0 ], [-5.0,-1.0 ] ]

# pts = polygon( vertices=pts, spacing=0.2, closed=True, gradients=0, at_edge=False )
pts = box( center=centers[0], height=5, width=5, angle=15, spacing=0.2, gradients=1, at_edge=True )

cbf = KernelBarrier(*initial_state, kernel=kernel, boundary=pts, centers=[centers[0]])
cbf.is_sos_convex(verbose=True)

# ----------------------------------------------- Plotting ----------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Invariant set plot for Kernel-based CLF-CBFs")
ax.set_aspect('equal', adjustable='box')

for pt in pts:
    coords = np.array(pt["coords"])
    ax.plot(coords[0], coords[1], 'k*', alpha=0.6)

    if "gradient" in pt.keys():
        gradient_vec = coords + np.array(pt["gradient"])
        ax.plot([ coords[0], gradient_vec[0]], [ coords[1], gradient_vec[1]], 'k-', alpha=0.6)

limits = (16*np.array([[-1, 1],[-1, 1]])).tolist()

ax.set_xlim(limits[0][0], limits[0][1])
ax.set_ylim(limits[1][0], limits[1][1])

num_lvls = 10
contour_boundary = cbf.plot_levels(levels = [ 0.5*(k - (num_lvls - 1))/(num_lvls - 1) for k in range(0,num_lvls,1) ], ax=ax, limits=limits)
# contour_boundary = cbf.plot_levels(levels = [ 2*k for k in range(0,num_lvls,1) ], ax=ax, limits=limits)

np.set_printoptions(precision=2, suppress=False)

plt.show()