import numpy as np
import scipy as sp
import sympy as sym

import matplotlib.pyplot as plt
from functions import Kernel, KernelLyapunov, KernelQuadratic, MultiPoly
from common import lyap, create_quadratic, rot2D, symmetric_basis

np.set_printoptions(precision=3, suppress=True)
limits = 3*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test Invex CLFs")
ax.set_aspect('equal', adjustable='box')

ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

# ------------------------------------ Define kernel and CLF -----------------------------------
n, d = 2, 2
kernel = Kernel(dim=n, degree=d)
print(kernel)
kernel_dim = kernel._num_monomials
A_list = kernel.get_A_matrices()

def quadratic(eigen: np.ndarray, angle: float):
    ''' Generates quadratic matrix '''
    if len(eigen) != 2: raise Exception("Only works for n= 2")
    if np.any(np.array(eigen) < 0): raise Exception("Eigenvalues must be non-negative.")
    R = rot2D(angle)
    return R.T @ np.diag(eigen) @ R

G = quadratic(eigen=[2, 1], angle=np.deg2rad(45))

N = sym.MatrixSymbol("N", n, kernel_dim)
N_list = [ N @ Ai for Ai in A_list ]    # list of n x p

M_list = []
for k in range(kernel_dim):
    Mk = np.zeros([n,n], dtype=N_list[0].dtype )
    for i, Ni in enumerate(N_list):
        Mk[:,i] = Ni[:,k]
    M_list.append(Mk)

# ------------------------------- Generation of ∇Φ(x) -----------------------------
delPhi = MultiPoly(kernel._powers, M_list).filter()
det = delPhi.determinant()

print(f"∇Φ(x) = {delPhi}")
print(f"|∇Φ|(x) = {det}")

# --------------------------- SOS factorization of |∇Φ(x)| -------------------------
sos_kernel = det.sos_kernel()
sos_index_matrix = det.sos_index_matrix(sos_kernel)
shape_matrix = det.shape_matrix(sos_kernel, sos_index_matrix)
shape_fun = sym.lambdify( N, shape_matrix )

def find_invex(N: np.ndarray):
    ''' Returns an N matrix that produces an invex function k(x).T N.T P N K(x) on the given kernel k(x) '''

    if N.shape != (n, kernel_dim):
        raise ValueError("N must be a n x p matrix.")

    shape_fun(N) >> 0

    P = N.T @ G @ N # this is always p.s.d.


# solve_N( kernel=kernel._powers, N=N )

# clf = KernelLyapunov(kernel=kernel, P=Pinit, limits=limits, spacing=0.01 )

# #---------------------------------------------------------------------------
# pt = plt.ginput(1, timeout=0)
# init_x = [ pt[0][0], pt[0][1] ]
# V = clf.function(init_x)
# clf_contour = clf.plot_levels(ax=ax, levels=[V])
# plt.pause(0.001)

# init_x_plot, = ax.plot([init_x[0]],[init_x[1]],'ob', alpha=0.5)
# N = 100
# for i in range(N):

#     pt = plt.ginput(1, timeout=0)
#     init_x = [ pt[0][0], pt[0][1] ]
#     init_x_plot.set_data([init_x[0]], [init_x[1]])

#     N = np.random.randn(n,kernel_dim)
#     P = N.T @ G @ N

#     clf.set_params(P=P)
#     clf.generate_contour()

#     if "clf_contour" in locals():
#         for coll in clf_contour:
#             coll.remove()
#         del clf_contour

#     U, sdvals, V = np.linalg.svd(N)

#     print(f"U(N) = {U}")
#     print(f"Singular values of N = {sdvals}")
#     print(f"V(N) = {V}")

#     # print(F"P = {P}")
#     print(f"λ(P) = {np.linalg.eigvals(P)}")
#     V = clf.function(init_x)
#     print(f"V({init_x}) = {V}")

#     num_levels = 20
#     clf_contour = clf.plot_levels(ax=ax, levels=[ V*((k+1)/num_levels) for k in range(num_levels) ])
#     plt.pause(0.001)

# plt.show()