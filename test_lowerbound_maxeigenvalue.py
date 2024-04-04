import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from sympy import MatrixSymbol
from functions import Kernel, KernelLyapunov
from common import symmetric_basis, create_quadratic, rot2D

n = 2
initial_state = [ np.random.rand() for _ in range(n) ]

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test lowerbound on max λ(Hv)")
ax.set_aspect('equal', adjustable='box')
limits = 10*np.array([[-1, 1],[-1, 1]])

ax.set_xlim(limits[0][0], limits[0][1])
ax.set_ylim(limits[1][0], limits[1][1])

kernel = Kernel(*initial_state, degree=3)
p = kernel.kernel_dim
Amatrices = kernel.get_A_matrices()

Asum = sum(Amatrices)
Asum2 = Asum @ Asum

Psym = MatrixSymbol('P', p, p)

def non_nsd_Hessian_constr(Pvar):
    return Asum.T @ Pvar @ Asum + Asum.T @ Asum.T @ Pvar

def extract_upper_block(Hsym):

    for i in range(0,Hsym.shape[0]):
        half_down = Hsym[i:, :]
        if np.all( half_down == 0 ): break

    if np.all( half_down == 0 ):
        return ( slice(0,i), slice(0,i) )
    
    return ( slice(0,Hsym.shape[0]), slice(0,Hsym.shape[1]) )

print( non_nsd_Hessian_constr(Psym).shape )

lowerbound_shape = extract_upper_block(non_nsd_Hessian_constr(Psym))
print( non_nsd_Hessian_constr(Psym)[lowerbound_shape].shape )

Pnom = cp.Parameter( (p,p), symmetric=True )
Pvar = cp.Variable( (p,p), symmetric=True )
cost = cp.norm(Pvar - Pnom)

# constraints = [ Pvar >> 0, non_nsd_Hessian_constr(Pvar) >> 0 ]
constraints = [ Pvar >> 0, non_nsd_Hessian_constr(Pvar)[lowerbound_shape] >> 0 ]

problem = cp.Problem( cp.Minimize( cost ), constraints )

N = 10000
for k in range(N):

    pt = plt.ginput(1, timeout=0)
    x = [ pt[0][0], pt[0][1] ]

    # Pnom.value = sum([ np.random.randn()*B for B in symmetric_basis(p) ])

    Psqrt = np.random.randn(p,p)
    Pnom.value = Psqrt.T @ Psqrt

    problem.solve(verbose=True)

    Pnom_eigs = np.linalg.eigvals(Pnom.value)

    print("------------------------- Pnom -------------------------")
    print(f"λ(Pnom) = {Pnom_eigs}")
    print(f"λ(Pnom_Hessian) = {np.linalg.eigvals(non_nsd_Hessian_constr(Pnom.value))}")

    P = Pvar.value
    P_eigs = np.linalg.eigvals(P)

    print("------------------------- P -------------------------")
    print(f"λ(P) = {P_eigs}")
    print(f"λ(P_Hessian) = {np.linalg.eigvals(non_nsd_Hessian_constr(P))}\n")

    clf = KernelLyapunov(*initial_state, kernel=kernel, P=P)

    if "clf_contour" in locals():
        for coll in clf_contour:
            coll.remove()
        del clf_contour

    V = clf.function(x)
    clf_contour = clf.plot_levels(levels=[V], ax=ax, limits=limits.tolist(), spacing=0.1)
    plt.pause(10e-4)