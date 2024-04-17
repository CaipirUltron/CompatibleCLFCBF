import numpy as np
import sympy as sym
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt

from functions import Kernel, KernelQuadratic, LeadingShape, KernelLyapunov, KernelBarrier
from common import create_quadratic, rot2D, box, polygon, lyap

np.set_printoptions(precision=4, suppress=True)

limits = 12*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test level sets")
ax.set_aspect('equal', adjustable='box')

xmin, xmax, ymin, ymax = limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
d=3
kernel = Kernel(dim=2, degree=d)
print(kernel)
kernel_dim = kernel._num_monomials

As = kernel.Asum
As2 = As @ As
Ip = np.eye(kernel_dim)
kron = np.kron(As2.T, Ip) + np.kron(Ip, As2.T)

P_var = cp.Variable( (kernel_dim, kernel_dim), symmetric=True )
Pnom_var = cp.Parameter( (kernel_dim, kernel_dim), symmetric=True )

n = 3

Zeros11 = np.zeros((n, n))
Zeros22 = np.zeros((kernel_dim-n, kernel_dim-n))
Zeros12 = np.zeros((n, kernel_dim-n))

I1 = np.eye(n)
I2 = np.eye(kernel_dim-n)

P11_var = cp.Variable( (n,n), symmetric=True )
P12_var = cp.Variable( (n, kernel_dim-n) )
P22_var = cp.Variable( (kernel_dim-n,kernel_dim-n), symmetric=True )

Pfiltered_var = cp.bmat([ [Zeros11  , P12_var ], 
                          [P12_var.T, P22_var ] ])

Pnull_var = cp.bmat([ [P11_var  , Zeros12 ], 
                      [Zeros12.T, Zeros22 ] ])

P_var = Pfiltered_var + Pnull_var

L11_var = cp.Variable( (n,n), symmetric=True )
L12_var = cp.Variable( (n, kernel_dim-n) )
L22_var = cp.Variable( (kernel_dim-n,kernel_dim-n), symmetric=True )

L_var = cp.bmat([ [L11_var  , L12_var ], 
                  [L12_var.T, Zeros22 ] ])

Epsilon = np.diag(4*np.random.rand(kernel_dim))
Epsilon = Epsilon.T @ Epsilon

#------------ This works (impossible to give Pnom) ---------------
# cost = cp.norm( kron @ cp.vec(P_var) - cp.vec(L_var) )
# constraints = [ Pnom_var >> P_var, P_var >> 0, L_var >> 0 ]

#---- This also works (but restricts P to be partially zero) -----
cost = cp.norm( P_var - Pnom_var )
constraints = [ P_var >> 0 ]
constraints += [ lyap((As2).T, P_var) >> 0 ]
# constraints += [ lyap(As.T, lyap(As.T, P_var)) >> 0 ]
constraints += [ P12_var[:,:] == 0 ]
#-----------------------------------------------------------------
# cost = cp.norm( P_var - Pnom_var )
# constraints = [ P_var >> 0 ]
# constraints += [ L12_var == Zeros12 ]
# constraints += [ lyap((As2).T, P_var) == L_var ]

problem = cp.Problem( cp.Minimize(cost), constraints )

P = np.random.randn(kernel_dim, kernel_dim)

# P = P.T @ P

Pnom_var.value = P.T @ P
problem.solve(verbose=True, max_iters=20000)
P = P_var.value

R = lyap(As.T, lyap(As.T, P))
L = lyap(( As2 ).T, P)
M = lyap(( As2 - Epsilon ).T, P)

L11 = L[0:n,0:n]

print(f"M = \n {M}")
print(f"P = \n {P}")
print(f"Pfiltered = \n {Pfiltered_var.value}")
print(f"Pnull = \n {Pnull_var.value}")

print(f"R = \n {R}")
print(f"λ(R) = \n{np.linalg.eigvals(R)}")

AtPA = As.T @ P @ As

print(f"As.T P A = \n{AtPA}")

# print(f"L_var = \n{L_var.value}")
print(f"L = \n{L}")
print(f"λ(L) = \n{np.linalg.eigvals(L)}")

print(f"λ(L11) = {np.linalg.eigvals(L11)}")

print(f"λ(P) = {np.linalg.eigvals(P)}")