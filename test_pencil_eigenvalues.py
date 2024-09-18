import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from functions import Kernel

# np.set_printoptions(precision=3, suppress=True)
limits = 12*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test Invex CLFs")
ax.set_aspect('equal', adjustable='box')

ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

# ------------------------------------ Define kernel -----------------------------------
n, d = 2, 2
kernel = Kernel(dim=n, degree=d)
print(kernel)
kernel_dim = kernel._num_monomials
q = kernel.dim_det_kernel
p = kernel._num_monomials

print(kernel.det_kernel)

#-------------------------------------- Define pencil ----------------------------------
A = np.random.randn(n,p)
B = np.random.randn(n,p)
AA, BB, alpha, beta, Q, Z = sp.linalg.ordqz( A @ A.T, B @ A.T , output='real' )

print(f"AA = {AA}")
print(f"BB = {BB}")
print(f"unitary error of Q = {np.linalg.norm(np.eye(n) - Q.T @ Q)}")
print(f"unitary error of Z = {np.linalg.norm(np.eye(n) - Z.T @ Z)}")

print(f"(A - Q AA Z') error = {np.linalg.norm( A @ A.T - Q @ AA @ Z.T)}")
print(f"(B - Q BB Z') error = {np.linalg.norm( B @ A.T - Q @ BB @ Z.T)}")

print(f"alpha = {alpha.real}")
print(f"beta = {beta}")

lamb = np.random.randn()
P = lamb * A - B
NULLright = sp.linalg.null_space(P)
print(f"NULLSPACE = {NULLright}")

# for k in range(n):
#     P = alpha[k].real * (A @ A.T) - beta[k] * (B @ A.T)
#     NULLright = sp.linalg.null_space(P)
#     print(f"(alpha A - beta B) NULLright = \n{ P @ NULLright }")