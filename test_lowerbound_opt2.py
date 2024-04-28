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
kernel = Kernel(dim=3, degree=3)
print(kernel)
kernel_dim = kernel._num_monomials

As = kernel.Asum
As2 = As @ As

Pnom_var = cp.Parameter( (kernel_dim, kernel_dim), symmetric=True )

n = kernel._dim
d = kernel._degree
blk_sizes = kernel._block_sizes

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

Pnn_var = cp.Variable( (n+1,n+1), symmetric=True )
if r > n+1:
    P1r_var = cp.Variable( (n+1,r-n-1) )
    Prr_var = cp.Variable( (r-n-1, r-n-1), symmetric=True )

    Z = np.zeros((n+1,r-n-1))
    Znn = np.zeros((n+1,n+1))
    Zrr = np.zeros((r-n-1,r-n-1))

    P11_var = cp.bmat([ [ Pnn_var ,  Z  ], 
                        [   Z.T   , Zrr ] ])
    Pbar11_var = cp.bmat([ [   Znn     , P1r_var ], 
                           [ P1r_var.T , Prr_var ] ])
else:
    P11_var = Pnn_var
    Pbar11_var = np.zeros((n+1,n+1))

P22_var = cp.Variable( (blk_sizes[1],blk_sizes[1]), symmetric=True )
P33_var = cp.Variable( (blk_sizes[2],blk_sizes[2]), symmetric=True )

P12_var = cp.Variable( (blk_sizes[0],blk_sizes[1]) )
P13_var = cp.Variable( (blk_sizes[0],blk_sizes[2]) )
P23_var = cp.Variable( (blk_sizes[1],blk_sizes[2]) )

Pl_var = cp.bmat([ [P11_var  , Zeros12  , Zeros13 ], 
                   [Zeros12.T, P22_var  , P23_var ],
                   [Zeros13.T, P23_var.T, P33_var ] ])

Pr_var = cp.bmat([ [Pbar11_var, P12_var  , P13_var ], 
                   [P12_var.T , Zeros22  , Zeros23 ],
                   [P13_var.T , Zeros23.T, Zeros33 ] ])

P_var = Pl_var + Pr_var

L11_var = cp.Variable( (blk_sizes[0],blk_sizes[0]), symmetric=True )
L22_var = cp.Variable( (blk_sizes[1],blk_sizes[1]), symmetric=True )
L33_var = cp.Variable( (blk_sizes[2],blk_sizes[2]), symmetric=True )

L12_var = cp.Variable( (blk_sizes[0],blk_sizes[1]) )
L13_var = cp.Variable( (blk_sizes[0],blk_sizes[2]) )
L23_var = cp.Variable( (blk_sizes[1],blk_sizes[2]) )

L_var = cp.bmat([ [L11_var  , L12_var  , L13_var ], 
                  [L12_var.T, L22_var  , L23_var ],
                  [L13_var.T, L23_var.T, L33_var ] ])

clf = KernelLyapunov(kernel=kernel, P=np.zeros([kernel_dim, kernel_dim]), limits=limits)

#------------ This works (impossible to give Pnom) ---------------
# cost = cp.norm( kron @ cp.vec(P_var) - cp.vec(L_var) )
# constraints = [ Pnom_var >> P_var, P_var >> 0, L_var >> 0 ]

#---- This also works (but restricts P to be partially zero according to block pattern) -----
cost = cp.norm( P_var - Pnom_var )
constraints = [ Pl_var >> 0 ]
constraints += [ lyap(As2.T, P_var) == 0 ]
# constraints += [ lyap(As2.T, Pr_var) + 2 * As.T @ P_var @ As >> 0 ]
# constraints += [ lyap(As2.T, Pr_var) == 0 ]

constraints += [ P12_var == 0, P13_var == 0 ]
if r > n+1: constraints += [ P1r_var == 0 , Prr_var >> 0 ]

constraints += [ cp.lambda_max(P_var) <= 100 ]
#---------------------------------------------------------------------------------------------
prob = cp.Problem( cp.Minimize(cost), constraints )

while True:

    pt = plt.ginput(1, timeout=0)
    x = [ pt[0][0], pt[0][1] ]

    # center = np.zeros(2)
    # clf_eig = np.random.randint(low=1, high=10, size=2)
    # clf_angle = np.random.randint(low=0, high=180)
    # clf_center = np.random.randn(2) + center
    # Pnom_var.value = create_quadratic( eigen=clf_eig, R=rot2D(np.deg2rad(clf_angle)), center=clf_center, kernel_dim=kernel_dim )

    P = np.zeros([kernel_dim, kernel_dim])
    for basis in symmetric_basis(kernel_dim): P += (100*np.random.randn() + 50)*basis
    Pnom_var.value = P

    # P = np.random.randn(kernel_dim, kernel_dim)
    # Pnom_var.value = P.T @ P

    prob.solve(solver="SCS",verbose=True, max_iters=50000)

    Pl = Pl_var.value
    Pr = Pr_var.value

    P = P_var.value
    L = lyap(As2.T, P)
    R = 2* As.T @ P @ As
    M = lyap(As.T, lyap(As.T, P))

    print(f"Kernel block sizes = {blk_sizes}")
    if sum(blk_sizes) != kernel_dim: raise Exception("Block sizes are not correctly defined.")

    print(f"P = \n {P}")
    print(f"Pl = \n {Pl}")
    print(f"Pr = \n {Pr}")

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
    levels = [ k*step for k in range(num_levels)]
    clf_contour = clf.plot_levels(levels=levels, ax=ax, limits=limits, spacing=0.1)
    plt.pause(1e-4)