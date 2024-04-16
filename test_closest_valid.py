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
kernel_dim = kernel._num_monomials

As = kernel.Asum
P_var = cp.Variable( (kernel_dim, kernel_dim), symmetric=True )
Pnom_var = cp.Parameter( (kernel_dim, kernel_dim), symmetric=True )

s01 = slice(0,3)
s12 = slice(3,6)
s23 = slice(6,kernel_dim)

Zeros11 = np.zeros((s01.stop-s01.start, s01.stop-s01.start))
Zeros22 = np.zeros((s12.stop-s12.start, s12.stop-s12.start))
Zeros33 = np.zeros((s23.stop-s23.start, s23.stop-s23.start))

Zeros12 = np.zeros((s01.stop-s01.start, s12.stop-s12.start))
Zeros13 = np.zeros((s01.stop-s01.start, s23.stop-s23.start))
Zeros23 = np.zeros((s12.stop-s12.start, s23.stop-s23.start))

I1 = np.eye(s01.stop-s01.start)
I2 = np.eye(s12.stop-s12.start)
I3 = np.eye(s23.stop-s23.start)

Q_var = lyap(As.T, P_var)

L11_var = cp.Variable( (s01.stop-s01.start, s01.stop-s01.start), symmetric=True )
L12_var = cp.Variable( (s01.stop-s01.start, s12.stop-s12.start) )
L13_var = cp.Variable( (s01.stop-s01.start, s23.stop-s23.start) )

L_var = cp.bmat([ [L11_var  , Zeros12  , Zeros13], 
                  [Zeros12.T, Zeros22  , Zeros23], 
                  [Zeros13.T, Zeros23.T, Zeros33] ])

R11_var = cp.Variable( (s01.stop-s01.start, s01.stop-s01.start), symmetric=True )
R22_var = cp.Variable( (s12.stop-s12.start, s12.stop-s12.start), symmetric=True )
R12_var = cp.Variable( (s01.stop-s01.start, s12.stop-s12.start) )
R13_var = cp.Variable( (s01.stop-s01.start, s23.stop - s23.start) )

# R_var = cp.bmat([ [R11_var  , R12_var  , R13_var], 
#                   [R12_var.T, R22_var  , Zeros23], 
#                   [R13_var.T, Zeros23.T, Zeros33] ])

R_var = cp.bmat([ [R11_var  , R12_var  , Zeros13], 
                  [R12_var.T, R22_var  , Zeros23], 
                  [Zeros13.T, Zeros23.T, Zeros33] ])

reducedR_var = cp.bmat([ [R11_var  , I2], 
                         [I2.T, Zeros22] ])
reducedRnom_var = cp.Parameter( (s12.stop, s12.stop), symmetric=True )

# R_var = cp.Variable( (kernel_dim, kernel_dim), symmetric=True )
# R_var = cp.bmat([ [R11_var  , Zeros12  , R13_var], 
#                   [Zeros12.T, R22_var  , Zeros23], 
#                   [R13_var.T, Zeros23.T, Zeros33] ])
# R_var = cp.bmat([ [R11_var  , Zeros12  , Zeros13], 
#                   [Zeros12.T, R22_var  , Zeros23], 
#                   [Zeros13.T, Zeros23.T, Zeros33] ])

# T_var = cp.bmat([[ -R11_var ,    I1   ], 
#                  [   I1   , Zeros11 ]])

n = 6
cost = cp.norm( P_var - Pnom_var )
# constraints = [ P_var >> 0, R_var[0:s12.stop, 0:s12.stop] >> 0 ]
constraints = [ P_var >> 0 ]
constraints += [ lyap((As@As).T, P_var) == L_var ]
# constraints += [ L12_var == Zeros12 ]
# constraints += [ L13_var == Zeros13 ]

# constraints += [ lyap(As.T, lyap(As.T, P_var)) == R_var ]
# constraints += [ lyap((As@As).T, P_var) == L_var ]
# constraints += [ 2 * As.T @ P_var @ As == R_var ]
# constraints += [ lyap((As@As).T, P_var) >> - 2 * As.T @ P_var @ As ]
# constraints += [ lyap((As@As).T, P_var) == R_var ]
# constraints += [ lyap(As.T, lyap(As.T, P_var))[s01,s01] == R11_var ]
# constraints += [ lyap(As.T, lyap(As.T, P_var))[s12,s12] == R22_var ]
# constraints += [ lyap(As.T, lyap(As.T, P_var))[s01,s23] == R13_var ]

rR = 5*np.random.randn(s12.stop, s12.stop)
reducedRnom_var.value = rR.T @ rR
problem = cp.Problem( cp.Minimize(cost), constraints )

P = 5*np.random.randn(kernel_dim, kernel_dim)
Pnom_var.value = P.T @ P

run = True
# run = False

if run:
    problem.solve(verbose=True, max_iters=30000)

P = P_var.value
Q = lyap(As.T, P)
R = lyap(As.T, lyap(As.T, P))

R11 = R[0:n,0:n]
# R22 = R[s12,s12]
# R[s01,s12] = Zeros12
# R13 = R[s01,s23]
# T = T_var.value

print(f"P = \n {P}")
print(f"Q = \n {Q}")
print(f"R = \n {R}")
print(f"N(R) = \n {sp.linalg.null_space(R)}")

# print(f"L13 = \n{R13}")
# print(f"Eigenvals of L13 L13.T: {np.linalg.eigvals(R13 @ R13.T)}")

AtPA = As.T @ P @ As
Lyap2 = lyap((As @ As).T, P)

print(f"As.T P A = \n{AtPA}")
print(f"As2.T P + P As2 = \n{Lyap2}")

print(f"λ(R11) = {np.linalg.eigvals(R11)}")
print(f"N(R11) = \n {sp.linalg.null_space(R11)}")

print(f"λ(P) = {np.linalg.eigvals(P)}")
print(f"λ(R) = {np.linalg.eigvals(R)}")
# print(f"λ(T) = {np.linalg.eigvals(T)}")

def polyeig(*A):
    """
    Solve the polynomial eigenvalue problem:
        (A0 + e A1 +...+  e**p Ap)x=0 

    Return the eigenvectors [x_i] and eigenvalues [e_i] that are solutions.

    Usage:
        X,e = polyeig(A0,A1,..,Ap)

    Most common usage, to solve a second order system: (K + C e + M e**2) x =0
        X,e = polyeig(K,C,M)

    """
    if len(A)<=0:
        raise Exception('Provide at least one matrix')
    for Ai in A:
        if Ai.shape[0] != Ai.shape[1]:
            raise Exception('Matrices must be square')
        if Ai.shape != A[0].shape:
            raise Exception('All matrices must have the same shapes')

    n = A[0].shape[0]
    l = len(A)-1 
    # Assemble matrices for generalized problem
    C = np.block([
        [np.zeros((n*(l-1),n)), np.eye(n*(l-1))],
        [-np.column_stack( A[0:-1])]
        ])
    D = np.block([
        [np.eye(n*(l-1)), np.zeros((n*(l-1), n))],
        [np.zeros((n, n*(l-1))), A[-1]          ]
        ])
    # Solve generalized eigenvalue problem
    e, X = sp.linalg.eig(C, D)
    if np.all(np.isreal(e)):
        e=np.real(e)
    X=X[:n,:]

    # Sort eigenvalues/vectors
    #I = np.argsort(e)
    #X = X[:,I]
    #e = e[I]

    # Scaling each mode by max
    X /= np.tile(np.max(np.abs(X),axis=0), (n,1))

    return X, e


# X, e = polyeig(-R13 @ R13.T, -R11, I1)

# print(f"Quadratic eigenvalues = {e}")