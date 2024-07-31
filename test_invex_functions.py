import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from functions import Kernel, KernelLyapunov
from common import hessian_2Dquadratic, PSD_closest, NSD_closest, timeit

# np.set_printoptions(precision=3, suppress=True)
limits = 3*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test Invex CLFs")
ax.set_aspect('equal', adjustable='box')

ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

# ------------------------------------ Define kernel and CLF -----------------------------------
n, d = 2, 3
kernel = Kernel(dim=n, degree=d)
print(kernel)
kernel_dim = kernel._num_monomials
q = kernel.dim_det_kernel

num_sim = 5000
for i in range(num_sim):

    N = np.random.randn(n,kernel_dim)
    eigsD = np.linalg.eigvals( kernel.D(N) )

    if np.all(eigsD > 0):
        print(f"Structurally PSD!")
        print(f"Eigs of D(N) = {eigsD}")

    if np.all(eigsD < 0):
        print(f"Structurally NSD!")
        print(f"Eigs of D(N) = {eigsD}")

G = hessian_2Dquadratic(eigen=[2, 1], angle=np.deg2rad(45))

tol = 1e-1
Tol = np.zeros((q,q))
Tol[0,0] = 1
Tol = tol*Tol

#---------------------------------- Function for invexification -----------------------------------
# @timeit
def find_invex(Ninit: np.ndarray):
    '''
    Returns an N matrix that produces an invex function k(x).T N.T P N K(x) on the given kernel k(x).
    PROBLEM: find N such that shape_fun(N) >> 0
    '''
    if Ninit.shape != (n, kernel_dim):
        raise ValueError("N must be a n x p matrix.")

    Dinit = kernel.D(Ninit)
    dist_to_psd = np.linalg.norm( PSD_closest(Dinit) - Dinit )
    dist_to_nsd = np.linalg.norm( NSD_closest(Dinit) - Dinit )

    if dist_to_psd <= dist_to_nsd: 
        cone = +1
        print(f"D(Ninit) closer to PSD cone.")
    else: 
        cone = -1
        print(f"D(Ninit) closer to NSD cone.")

    def objective(var):
        N = var.reshape((n, kernel_dim))
        D = kernel.D(N)

        if cone == +1:
            Proj = PSD_closest(D)
        if cone == -1:
            Proj = NSD_closest(D)

        # return np.linalg.norm(N - Ninit)
        return np.linalg.norm(Proj)
    
    def invex(var: np.ndarray):
        N = var.reshape((n, kernel_dim))
        D = kernel.D(N)

        if cone == +1:
            return min(np.linalg.eigvals( D - Tol ))
        if cone == -1:
            return -max(np.linalg.eigvals( D + Tol ))
        
    def centered(var: np.ndarray):
        N = var.reshape((n, kernel_dim))
        m0 = kernel.function(np.zeros(n))
        return m0.T @ N.T @ G @ N @ m0

    def orthonormality_constr(var: np.ndarray) -> float:
        ''' Keeps N orthonormal '''

        N = var.reshape((n, kernel_dim))
        return np.linalg.norm( N @ N.T - np.eye(n) )

    constraints = []
    # constraints += [ {"type": "ineq", "fun": invex} ]
    constraints += [ {"type": "eq", "fun": centered} ]
    constraints += [ {"type": "eq", "fun": orthonormality_constr} ]

    init_var = Ninit.flatten()
    sol = minimize( objective, init_var, constraints=constraints, options={"disp": False, "maxiter":1000} )

    print(sol.message)
    N = sol.x.reshape((n, kernel_dim))

    print(f"Invexity of N = {objective(sol.x)}")
    print(f"Orthonormality of N = {orthonormality_constr(sol.x)}")

    return N, cone

# ----- Example of N with unimodular |∇Φ(x)| = 1 resulting in CONVEX function (quadratic) -----
Nconvex = np.zeros((n, kernel_dim))
Nconvex[:,1:n+1] = np.eye(n)

Pconvex = Nconvex.T @ G @ Nconvex
clf = KernelLyapunov(kernel=kernel, P=Pconvex, limits=limits, spacing=0.01 )

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
    # Ninit = np.random.randint(low=1, high=5, size=(n,kernel_dim))
    # print(f"Coefficients of |∇Φ(x)| at Ninit = {det_coeffs_fun(Ninit)}")

    N, cone = find_invex(Ninit)
    G = np.random.randn(n,n)
    G = G.T @ G

    D = kernel.D(N)
    Deigs = np.linalg.eigvals(D)
    if np.any(cone*Deigs < 0):
        if cone > 0: warnings.warn("Negative eigenvalues found in D(N).")
        else: warnings.warn("Positive eigenvalues found in D(N).")

    P = N.T @ G @ N

    clf.set_params(P=P)
    clf.generate_contour()

    if "clf_contour" in locals():
        for coll in clf_contour:
            coll.remove()
        del clf_contour

    V = clf.function(init_x)
    # print(f"V(x) = {V}")
    print(f"λ( D(N) ) = {Deigs}")

    num_levels = 20
    clf_contour = clf.plot_levels(ax=ax, levels=[ V*((k+1)/num_levels) for k in range(num_levels) ])
    plt.pause(0.001)