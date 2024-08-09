import itertools
import numpy as np
import matplotlib.pyplot as plt

from common import box, discretize, segmentize
from functions import Kernel, InvexProgram, KernelLyapunov, KernelBarrier
from shapely import LineString

# np.set_printoptions(precision=3, suppress=True)
limits = 12*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test Invex CLFs")
ax.set_aspect('equal', adjustable='box')

ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

# ------------------------------------ Define kernel -----------------------------------
n, d = 2, 3
kernel = Kernel(dim=n, degree=d)
print(kernel)
kernel_dim = kernel._num_monomials
q = kernel.dim_det_kernel
p = kernel._num_monomials

#---------------------------- Define some points for fitting ---------------------------
''' Box-shaped obstacle (convex) '''
center = [ 3.0, -2.0 ]
box_angle = 30
box_height, box_width = 5, 8
boundary_pts = box( center=center, height=box_height, width=box_width, angle=box_angle, spacing=0.4 )

''' U-shaped obstacle (non-convex) '''
# center = (0, 0)
# centers = [(-4, 3), (-4, 0), center, (4, 0), (4, 3)]
# skeleton_line = LineString(centers)
# obstacle_poly = skeleton_line.buffer(1.0, cap_style='flat')
# boundary_pts = discretize(obstacle_poly, spacing=0.4)

points = [ {"point": center, "level": -0.5} ]
for pt in boundary_pts:
    coords = np.array(pt)
    print(coords)
    ax.plot(coords[0], coords[1], 'k*', alpha=0.6)
    points.append({"point": pt, "level": 0.0})

#------------------------------------- Compute invex -----------------------------------
invex_program = InvexProgram( kernel, fit_to='cbf', points=points, center=center, mode='invexcost', 
                              slack_gain=1e-0, invex_gain=1e+2, cost_gain=1e+2, invex_tol=0.0 )

Q = invex_program.solve_program()
cbf = KernelBarrier(kernel=kernel, Q=Q, limits=limits, spacing=0.2 )

#---------------------------------------- Plotting -------------------------------------
num_levels=10
while True: 

    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    h = cbf.function(init_x)
    print(f"h(x) = {h}")

    if "cbf_contour" in locals():
        for coll in cbf_contour:
            coll.remove()
        del cbf_contour

    cbf_contour = cbf.plot_levels(ax=ax, levels=[ h*((k+1)/num_levels) for k in range(num_levels) ])
    plt.pause(0.01)