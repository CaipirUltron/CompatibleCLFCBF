import itertools
import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt

from functions import Kernel, KernelBarrier

# np.set_printoptions(precision=3, suppress=True)
limits = 12*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("SDP Invex CLF-CBFs")
ax.set_aspect('equal', adjustable='box')

ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

# ------------------------------------ Define kernel -----------------------------------
n, d = 2, 3
kernel = Kernel(dim=n, degree=d)
print(kernel)
A_list = kernel.get_A_matrices()
kernel_dim = kernel._num_monomials
q = kernel.dim_det_kernel
p = kernel._num_monomials
r = kernel._jacobian_dim

print(kernel.det_kernel)

#----------------------------------- Define center point -------------------------------
center = np.array([ 1.0, 5.0 ])
mc = kernel.function(center)

E = np.eye(p)
Matrix = np.zeros((p, p-n))
for dim in range(p-n):
    if dim == 0:
        Matrix[:,dim] = mc
    else:
        Matrix[:,dim] = E[:,p-dim]
N = sp.linalg.null_space( Matrix.T ).T

print(N)
print(f"N mc = {N @ mc}")

Gnom = 1*np.eye(n)

Jphi_squared = np.block([[ mc.T @ Ai.T @ N.T @ Gnom @ N @ Aj @ mc for Ai in A_list ] for Aj in A_list ])
eigsJphi = np.linalg.eigvals(Jphi_squared)
print(f"Jphi = {eigsJphi}")

epsilon = min(eigsJphi)
C = epsilon*np.eye(n)
print(f"Jphi - eye = {np.linalg.eigvals(Jphi_squared-C)}")

N += 0e-6*np.random.randn(n,p)
R_blocks = [[ None for _ in range(n) ] for _ in range(n) ]
for i, Ai in enumerate(A_list):
    for j, Aj in enumerate(A_list):
        Cij = np.zeros((r,r))
        Cij[0,0] = C[i,j]
        R_blocks[i][j] = (Ai.T @ N.T @ G @ N @ Aj)[0:r,0:r] - Cij
R = cp.bmat(R_blocks)

cost = cp.norm( G - Gnom ,'fro')
problem = cp.Problem( cp.Minimize( cost ), constraints=[ R >> 0 ] )

try:
    problem.solve(verbose=True, solver=cp.CLARABEL)
    print(f"G = {G.value}")

    cbf = KernelBarrier(kernel=kernel, Q = N.T @ G.value @ N, limits=limits, spacing=0.1 )

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

except cp.SolverError as error:
    print(error)