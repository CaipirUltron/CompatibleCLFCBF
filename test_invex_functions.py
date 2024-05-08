import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from functions import Kernel, KernelLyapunov, MultiPoly
from common import rot2D

# np.set_printoptions(precision=3, suppress=True)
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

# print(f"|∇Φ(x)| = {det}")
print("Coefficients of |∇Φ(x)|:")
for k, c in enumerate(det.coeffs):
    print(f"c{k} = {c}")

print(type(det.coeffs))

det_coeffs_fun = sym.lambdify( N, det.coeffs )

# ----------- SOS factorization of |∇Φ(x)| (not needed for the idea of unimodular matrices) ---------
sos_kernel = det.sos_kernel()
sos_index_matrix = det.sos_index_matrix(sos_kernel)
shape_matrix = det.shape_matrix(sos_kernel, sos_index_matrix)
# sym.pprint(f"D(N) = {shape_matrix}")

# shape_fun = sym.lambdify( N, shape_matrix ) # >> 0

def find_invex(Ninit: np.ndarray):
    '''
    Returns an N matrix that produces an invex function k(x).T N.T P N K(x) on the given kernel k(x).
    PROBLEM: find N such that shape_fun(N) >> 0
    '''
    if Ninit.shape != (n, kernel_dim):
        raise ValueError("N must be a n x p matrix.")

    def objective(var):
        N = var.reshape((n, kernel_dim))
        return np.linalg.norm(N - Ninit)
        # return 1.0

    def unimodular(var: np.ndarray):
        N = var.reshape((n, kernel_dim))
        det = det_coeffs_fun(N)
        unimodular_target = np.zeros(len(det))
        unimodular_target[0] = 1
        return det - unimodular_target

    def unimodular_equality(var: np.ndarray):
        N = var.reshape((n, kernel_dim))
        zero_part = det_coeffs_fun(N)[1:]
        return zero_part - np.zeros(len(zero_part))

    def unimodular_inequality(var: np.ndarray):
        N = var.reshape((n, kernel_dim))
        return det_coeffs_fun(N)[0] - 1

    def centered(var: np.ndarray):
        N = var.reshape((n, kernel_dim))
        m0 = kernel.function(np.zeros(n))
        return m0.T @ N.T @ G @ N @ m0

    # tol = 5e-0
    # def invex(var: np.ndarray):
    #     N = var.reshape((n, kernel_dim))
    #     eigs = np.linalg.eigvals( shape_fun(N) )
    #     max_eig = float(np.max(eigs))
    #     min_eig = float(np.min(eigs))
    #     return + min_eig - tol

    # constraints = [ {"type": "ineq", "fun": invex} ]
    constraints = [ {"type": "eq", "fun": unimodular} ]
    # constraints = [ {"type": "eq", "fun": unimodular_equality} ]
    # constraints += [ {"type": "ineq", "fun": unimodular_inequality} ]
    constraints += [ {"type": "eq", "fun": centered} ]

    init_var = Ninit.flatten()
    sol = minimize( objective, init_var, constraints=constraints, options={"disp": True, "maxiter":1000} )

    print(sol.message)
    N = sol.x.reshape((n, kernel_dim))

    return N

# ----- Example of N with unimodular |∇Φ(x)| = 1 resulting in CONVEX function (quadratic) -----
Ninit = np.zeros((n, kernel_dim))
Ninit[:,1:n+1] = np.eye(n)

# ----- Example of N with unimodular |∇Φ(x)| = 1 resulting in invex, NON-CONVEX function (n=2, d=2) -----
# Ninit = np.array([[0, 1, 0, 1, -2, 1],
#                   [0, 0, 1, 1, -2, 1]])

print( Ninit )
print( det_coeffs_fun(Ninit) )

Pinit = Ninit.T @ G @ Ninit
# Dinit = shape_fun(Ninit)
# print(f"λ( D(N) ) = {np.linalg.eigvals(Dinit)}")
clf = KernelLyapunov(kernel=kernel, P=Pinit, limits=limits, spacing=0.01 )

#------------------------------ Plotting -----------------------------------
pt = plt.ginput(1, timeout=0)
init_x = [ pt[0][0], pt[0][1] ]
V = clf.function(init_x)
clf_contour = clf.plot_levels(ax=ax, levels=[V])
plt.pause(0.001)

init_x_plot, = ax.plot([init_x[0]],[init_x[1]],'ob', alpha=0.5)
num_sim = 100
for i in range(num_sim):

    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    init_x_plot.set_data([init_x[0]], [init_x[1]])

    Ninit = np.random.randn(n,kernel_dim)
    N = find_invex(Ninit)

    # D = shape_fun(N)
    P = N.T @ G @ N

    clf.set_params(P=P)
    clf.generate_contour()

    if "clf_contour" in locals():
        for coll in clf_contour:
            coll.remove()
        del clf_contour

    V = clf.function(init_x)
    # print(f"V({init_x}) = {V}")

    # print(f"λ(P) = {np.linalg.eigvals(P)}")
    # print(f"λ( D(N) ) = {np.linalg.eigvals(D)}")
    print(f"Coefficients of |∇Φ(x)| = {det_coeffs_fun(N)}")

    num_levels = 20
    clf_contour = clf.plot_levels(ax=ax, levels=[ V*((k+1)/num_levels) for k in range(num_levels) ])
    plt.pause(0.001)

plt.show()