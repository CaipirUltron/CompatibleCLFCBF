import itertools
import numpy as np
import scipy as sp
import cvxpy as cp
import sympy as sym
import matplotlib.pyplot as plt

from sympy import zeros, Matrix
from common import dirac, box, discretize, segmentize
from functions import Kernel, InvexProgram, KernelBarrier

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

# Null = sp.linalg.null_space( mc.reshape(1,-1) )
# S = np.random.randn(n, Null.shape[1])
# N = S @ Null.T

E = np.eye(p)
Matrix = np.zeros((p, p-n))
for dim in range(p-n):
    if dim == 0:
        Matrix[:,dim] = mc
    else:
        Matrix[:,dim] = E[:,p-dim]
N = sp.linalg.null_space( Matrix.T ).T

print(N.shape)
print(f"N mc = {N @ mc}")

# N = cp.Variable( (n,p) )
G = cp.Variable( (n,n), PSD=True )
# P = cp.Variable( (p,p), PSD=True )
P = N.T @ G @ N

C = np.zeros((r,r))
C[0,0] = center.T @ center
C[0,1:n+1] = -center
C[1:n+1,0] = -center
C[1:n+1,1:n+1] = 1*np.eye(n)

Gnom = 1*np.eye(n)
Jphi_sq_ref = np.block([[ (Ai.T @ N.T @ Gnom @ N @ Aj)[0:r,0:r] for i, Ai in enumerate(A_list) ] for j, Aj in enumerate(A_list) ])

# Jphi_sq_ref = np.block([[ (Ai.T @ N.T @ Gnom @ N @ Aj)[0:r,0:r] for i, Ai in enumerate(A_list) ] for j, Aj in enumerate(A_list) ])
print(f"Jphi_sq = {np.linalg.eigvals(Jphi_sq_ref).real}")

N += 0.01*np.random.randn(n,p)

# R = cp.bmat([[ (Ai.T @ N.T @ G @ N @ Aj)[0:r,0:r] - C * dirac(i,j) for i, Ai in enumerate(A_list) ] for j, Aj in enumerate(A_list) ])
R = cp.bmat([[ (Ai.T @ N.T @ G @ N @ Aj)[0:r,0:r] for i, Ai in enumerate(A_list) ] for j, Aj in enumerate(A_list) ]) - Jphi_sq_ref

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