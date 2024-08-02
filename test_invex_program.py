import numpy as np
import matplotlib.pyplot as plt

from common import box
from functions import Kernel, InvexProgram, KernelLyapunov, KernelBarrier

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

print(f"D(N) = \n{kernel.Dfun_symbolic}")

#---------------------------- Define some points for fitting ---------------------------
box_center = [ 0.0, 2.0 ]
box_angle = 30
box_height, box_width = 5, 5
boundary_pts = box( center=box_center, height=box_height, width=box_width, angle=box_angle, spacing=0.4 )

points = [ {"point": box_center, "level": -0.5} ]
for pt in boundary_pts:
    coords = np.array(pt)
    ax.plot(coords[0], coords[1], 'k*', alpha=0.6)
    points.append({"point": pt, "level": 0.0})

#------------------------------------- Compute invex -----------------------------------
N = 0.1*np.random.randn(p,p)
invex_program = InvexProgram(kernel, fit_to = 'cbf', points=points, initial_shape = N.T @ N, barrier_gain = 100, invex_tol=1e-1)
Q = invex_program.find_invex()

cbf = KernelBarrier(kernel=kernel, Q=Q, limits=limits, spacing=0.2 )

#---------------------------------------- Plotting -------------------------------------
while True:

    if "cbf_contour" in locals():
        for coll in cbf_contour:
            coll.remove()
        del cbf_contour

    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    h = cbf.function(init_x)
    print(f"h(x) = {h}")

    num_levels = 20
    cbf_contour = cbf.plot_levels(ax=ax, levels=[ h*((k+1)/num_levels) for k in range(num_levels) ])
    plt.pause(0.001)