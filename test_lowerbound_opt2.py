import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
from functions import Kernel, KernelLyapunov
from common import lyap, create_quadratic, rot2D, symmetric_basis

np.set_printoptions(precision=4, suppress=True)
limits = 12*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test CLF functions with lowerbound on max λ(Hv)")
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=2, degree=4)
print(kernel)
kernel_dim = kernel._num_monomials

As = kernel.Asum
As2 = As @ As

print(f"As2 = {As2}")

Pnom_var = cp.Parameter( (kernel_dim, kernel_dim), symmetric=True )

n = kernel._dim
d = kernel._degree
blk_sizes = kernel.blk_sizes 
sl_n, sl_r, sl_s, sl_t = kernel.sl_n, kernel.sl_r, kernel.sl_s, kernel.sl_t

clf = KernelLyapunov(kernel=kernel, P=np.zeros([kernel_dim, kernel_dim]), limits=limits)

# ---- This works (but restricts P to be partially zero according to block pattern) -----
L = kernel.get_left_lowerbound(clf.SHAPE)
L11 = L[ 0:sum(blk_sizes[0:2]), 0:sum(blk_sizes[0:2]) ]
L12 = L[sl_n, sl_s]
if sl_r.stop > sl_r.start: L12 = cp.bmat([ [ L12 ] , [ L[sl_r, sl_s] ] ])
L13 = L[sl_n, sl_t]
if sl_r.stop > sl_r.start: L13 = cp.bmat([ [ L13 ] , [ L[sl_r, sl_t] ] ])

R = kernel.get_right_lowerbound(clf.SHAPE)
R11 = R[ 0:sum(blk_sizes[0:2]), 0:sum(blk_sizes[0:2]) ]
R22 = R[sl_s,sl_s]
R12 = R[sl_n, sl_s]
if sl_r.stop > sl_r.start: R12 = cp.bmat([ [ R12 ] , [ R[sl_r, sl_s] ] ])

M = kernel.get_lowerbound(clf.SHAPE)
M00_22 = M[ 0:sum(blk_sizes[0:3]), 0:sum(blk_sizes[0:3]) ]

cost = cp.norm( clf.SHAPE - Pnom_var )

constraints = [ clf.SHAPE >> 0 ]
for col in As2.T:
    if np.any(col != 0.0):
        print(f"{col}")
        constraints += [ clf.SHAPE @ col == 0 ]
#---------------------------------------------------------------------------------------------
prob = cp.Problem( cp.Minimize(cost), constraints )

while True:

    pt = plt.ginput(1, timeout=0)
    x = [ pt[0][0], pt[0][1] ]

    # center = np.zeros(2)
    # clf_eig = np.array([10, 1])
    # clf_angle = -45
    # clf_center = 10*np.random.randn(2) + center
    # Pnom_var.value = create_quadratic( eigen=clf_eig, R=rot2D(np.deg2rad(clf_angle)), center=clf_center, kernel_dim=kernel_dim )
    # noise = np.zeros([kernel_dim, kernel_dim])
    # for basis in symmetric_basis(kernel_dim): noise += 0.01*np.random.randn()*basis
    # Pnom_var.value += noise.T @ noise

    P = np.zeros([kernel_dim, kernel_dim])
    for basis in symmetric_basis(kernel_dim): P += (np.random.randint(low=-50,high=50)*np.random.randn() + np.random.randint(low=-100,high=100))*basis
    Pnom_var.value = P

    # P = np.random.randn(kernel_dim, kernel_dim)
    # Pnom_var.value = P.T @ P

    prob.solve(solver="SCS",verbose=True, max_iters=10000)

    P = clf.SHAPE.value
    L = lyap(As2.T, P)
    R = 2* As.T @ P @ As
    M = lyap(As.T, lyap(As.T, P))

    print(f"Kernel block sizes = {blk_sizes}")
    if sum(blk_sizes) != kernel_dim: raise Exception("Block sizes are not correctly defined.")

    print(f"P = \n {P}")

    print(f"L = {L}")
    print(f"R = {R}")
    print(f"M = \n{M}")

    print(f"λ(P) = {np.linalg.eigvals(P)}")
    print(f"λ(L) = \n{np.linalg.eigvals(L)}")
    print(f"λ(R) = \n{np.linalg.eigvals(R)}")
    print(f"λ(M) = \n{np.linalg.eigvals(M)}")

    # Plots resulting CLF contour
    if "clf_contour" in locals():
        for coll in clf_contour:
            coll.remove()
        del clf_contour

    clf.set_params(P=P)

    num_levels = 10
    step = 100
    levels = [ (k**3)*step for k in range(num_levels)]
    clf_contour = clf.plot_levels(levels=levels, ax=ax, limits=limits, spacing=0.1)
    plt.pause(1e-4)