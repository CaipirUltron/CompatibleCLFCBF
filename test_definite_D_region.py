import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from common import box, discretize, segmentize
from functions import Kernel, InvexProgram, KernelBarrier
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

print(kernel.det_kernel)

#---------------------------- Define center point ---------------------------
center = [ 0.0, 10.0 ]
mc = kernel.function(center)
Null = sp.linalg.null_space( mc.reshape(1,-1) )

detected = False
Nshapes = 100000
for k in range(Nshapes):

    S = 100.0*np.random.randn( n, Null.shape[1] )
    N = S @ Null.T

    D = kernel.D(N)
    eigD = np.linalg.eigvals(D)

    if np.all(eigD >= 0.0):
        print(f"PSD detected!")
        detected = True
        break

    if np.all(eigD <= 0.0):
        print(f"NSD detected!")
        detected = True
        break

if detected:
    print(f"N mc = {N @ mc}")
    print(f"N = {N}")
    print(f"λ(D)(N) = {eigD}")

    cbf = KernelBarrier(kernel=kernel, Q = N.T @ N, limits=limits, spacing=0.2 )

    num_levels=10
    while True: 
        pt = plt.ginput(1, timeout=0)
        x = [ pt[0][0], pt[0][1] ]
        h = cbf.function(x)
        print(f"h = {h}")

        if "cbf_contour" in locals():
            for coll in cbf_contour:
                coll.remove()
            del cbf_contour

        cbf_contour = cbf.plot_levels(ax=ax, levels=[ h*((k+1)/num_levels) for k in range(num_levels) ])
        plt.pause(0.01)

else:
    print(f"No definite D(N) detected out of {Nshapes} random shapes with N mc = 0. ")