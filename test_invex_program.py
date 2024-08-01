import itertools
import warnings

import numpy as np
import matplotlib.pyplot as plt

from functions import Kernel, InvexProgram, KernelLyapunov

# np.set_printoptions(precision=3, suppress=True)
limits = 3*np.array((-1,1,-1,1))

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

#------------------------------------- Compute invex -----------------------------------
N = np.random.randn(p,p)
invex_program = InvexProgram(kernel, initial_shape = N.T @ N, invex_tol=1e-2)
P = invex_program.find_invex()

clf = KernelLyapunov(kernel=kernel, P=P, limits=limits, spacing=0.01 )

#---------------------------------------- Plotting -------------------------------------
while True:

    if "clf_contour" in locals():
        for coll in clf_contour:
            coll.remove()
        del clf_contour

    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    V = clf.function(init_x)
    print(f"V(x) = {V}")

    num_levels = 20
    clf_contour = clf.plot_levels(ax=ax, levels=[ V*((k+1)/num_levels) for k in range(num_levels) ])
    plt.pause(0.001)