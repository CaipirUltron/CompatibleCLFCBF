import numpy as np
import matplotlib.pyplot as plt

from functions import Kernel, KernelQuadratic, LeadingShape, KernelLyapunov, KernelBarrier
from common import create_quadratic, rot2D, box, polygon_shapely

limits = 12*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test level sets")
ax.set_aspect('equal', adjustable='box')

xmin, xmax, ymin, ymax = limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=2, degree=2)
kernel_dim = kernel._num_monomials

center = [ 0.0, 2.0 ]
P = 0.1*np.random.randn(kernel_dim, kernel_dim)
P = P.T @ P

vertices = [ (0,0), (1,0), (1,1), (0,1) ]

# pts = polygon_shapely(vertices, closed=True)
pts = box( center=center, height=5, width=5, angle=10, spacing=0.4 )

for pt in pts:
    coords = np.array(pt)
    ax.plot(coords[0], coords[1], 'k*', alpha=0.6)

fun = KernelBarrier(kernel=kernel, constant = 0.5, 
                      boundary=pts, centers=[center],
                      limits = limits, spacing = 0.1)

fun.plot_levels(ax=ax, levels = [0.0])
plt.show()