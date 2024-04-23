import numpy as np
import cvxpy as cp

from functions import Kernel
from common import lyap

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=2, degree=3)
print(kernel)
kernel_dim = kernel._num_monomials

As = kernel.Asum
As2 = As @ As
Ip = np.eye(kernel_dim)
kron = np.kron(As2.T, Ip) + np.kron(Ip, As2.T)

P_var = cp.Variable( (kernel_dim, kernel_dim), symmetric=True )
Pnom_var = cp.Parameter( (kernel_dim, kernel_dim), symmetric=True )

blk_sizes = kernel._block_sizes

n = kernel._dim
d = kernel._degree

r = max(n+1, blk_sizes[0])
Pslice_hor = slice(0,r)
Pslice_ver = slice(n+1, kernel_dim)
ZerosPslice = np.zeros([r, kernel_dim-n-1])

sl1 = slice(0, blk_sizes[0])
sl2 = slice(blk_sizes[0], blk_sizes[0] + blk_sizes[1])
sl3 = slice(blk_sizes[0] + blk_sizes[1], blk_sizes[0] + blk_sizes[1] + blk_sizes[2])

Zeros11 = np.zeros((blk_sizes[0],blk_sizes[0]))
Zeros22 = np.zeros((blk_sizes[1],blk_sizes[1]))
Zeros33 = np.zeros((blk_sizes[2],blk_sizes[2]))

Zeros12 = np.zeros((blk_sizes[0],blk_sizes[1]))
Zeros13 = np.zeros((blk_sizes[0],blk_sizes[2]))
Zeros23 = np.zeros((blk_sizes[1],blk_sizes[2]))

I1 = np.eye(blk_sizes[0])
I2 = np.eye(blk_sizes[1])
I3 = np.eye(blk_sizes[2])

P11_var = cp.Variable( (blk_sizes[0],blk_sizes[0]), symmetric=True )
P22_var = cp.Variable( (blk_sizes[1],blk_sizes[1]), symmetric=True )
P33_var = cp.Variable( (blk_sizes[2],blk_sizes[2]), symmetric=True )

P12_var = cp.Variable( (blk_sizes[0],blk_sizes[1]) )
P13_var = cp.Variable( (blk_sizes[0],blk_sizes[2]) )
P23_var = cp.Variable( (blk_sizes[1],blk_sizes[2]) )

P_var = cp.bmat([ [P11_var  , P12_var  , P13_var ], 
                  [P12_var.T, P22_var  , P23_var ],
                  [P13_var.T, P23_var.T, P33_var ] ])

L11_var = cp.Variable( (blk_sizes[0],blk_sizes[0]), symmetric=True )
L22_var = cp.Variable( (blk_sizes[1],blk_sizes[1]), symmetric=True )
L33_var = cp.Variable( (blk_sizes[2],blk_sizes[2]), symmetric=True )

L12_var = cp.Variable( (blk_sizes[0],blk_sizes[1]) )
L13_var = cp.Variable( (blk_sizes[0],blk_sizes[2]) )
L23_var = cp.Variable( (blk_sizes[1],blk_sizes[2]) )

L_var = cp.bmat([ [L11_var  , L12_var  , L13_var ], 
                  [L12_var.T, L22_var  , L23_var ],
                  [L13_var.T, L23_var.T, L33_var ] ])

P = np.random.randn(kernel_dim, kernel_dim)
Pnom_var.value = P.T @ P

#------------ This works (impossible to give Pnom) ---------------
# cost = cp.norm( kron @ cp.vec(P_var) - cp.vec(L_var) )
# constraints = [ Pnom_var >> P_var, P_var >> 0, L_var >> 0 ]

#---- This also works (but restricts P to be partially zero according to block pattern) -----
cost = cp.norm( P_var - Pnom_var )
constraints = [ P_var >> 0 ]
constraints += [ P_var[Pslice_hor, Pslice_ver] == ZerosPslice ]
# constraints += [ lyap(As.T, lyap(As.T, P_var)) >> 0 ]
constraints += [ lyap(As2.T, P_var) == 0 ]
#---------------------------------------------------------------------------------------------
prob = cp.Problem( cp.Minimize(cost), constraints )
prob.solve(solver="SCS",verbose=True, max_iters=100000)

P = P_var.value
L = lyap(As2.T, P)
R = 2* As.T @ P @ As
M = lyap(As.T, lyap(As.T, P))

print(f"Kernel block sizes = {blk_sizes}")
if sum(blk_sizes) != kernel_dim: raise Exception("Block sizes are not correctly defined.")

print(f"P = \n {P}")
print(f"L = {L}")
print(f"L = {R}")
print(f"M = \n{M}")

print(f"λ(P) = {np.linalg.eigvals(P)}")
print(f"λ(L) = \n{np.linalg.eigvals(L)}")
print(f"λ(R) = \n{np.linalg.eigvals(R)}")
print(f"λ(M) = \n{np.linalg.eigvals(M)}")