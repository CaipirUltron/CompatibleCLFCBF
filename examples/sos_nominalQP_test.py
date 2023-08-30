import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP

initial_state = [-4.2, -5.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
max_degree = 2
kernel = Kernel(*initial_state, degree = max_degree)
p = kernel.kernel_dim
print(kernel)
print(kernel.alpha)

# -------------------------------------------------- Define system ---------------------------------------------------------
F = np.zeros([p,p])
def g(state):
    return np.eye(m)
plant = ConservativeAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

clf_eig = np.flip([5.0, 1.0])
clf_center = np.flip([-4.0, -4.0])
P = np.zeros([p,p])
P[0,0] = clf_center.T @ np.diag(clf_eig) @ clf_center
for k in range(n):
    P[0,k+1] = -clf_center[k]*clf_eig[k]
    P[k+1,0] = -clf_center[k]*clf_eig[k]
    P[k+1,k+1] = clf_eig[k]

cbf_eig = np.flip([1.0, 4.0])
cbf_center = np.flip([0.0, 0.0])
Q = np.zeros([p,p])
Q[0,0] = cbf_center.T @ np.diag(cbf_eig) @ cbf_center
for k in range(n):
    Q[0,k+1] = -cbf_center[k]*cbf_eig[k]
    Q[k+1,0] = -cbf_center[k]*cbf_eig[k]
    Q[k+1,k+1] = cbf_eig[k]

Proot = 0.1*np.random.rand(p,p)
P = Proot.T @ Proot
clf_center = np.array([4.0, 5.0])

# Qroot = 0.1*np.random.rand(p,p)
# Q = Qroot.T @ Qroot
boundary_points = np.array([ [-4.0, 0.0], [-4.0, -1.0], [2.0, 0.5], [4.0, -1.0], [4.0, 0.0] ])

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
clf = KernelLyapunov(*initial_state, kernel=kernel, P=P)
clf.define_center( clf_center )

# ---------------------------------------------------- Define CBF ----------------------------------------------------------
cbf = KernelBarrier(*initial_state, kernel=kernel, Q=Q)
cbf.define_boundary( boundary_points )

cbfs = [cbf]
from controllers import find_nearest_boundary, compute_equilibria_algorithm7

# equilibrium = compute_equilibria_algorithm7(plant, clf, cbf, initial_state, c = 1)
# print(equilibrium)

# boundary_pt = find_nearest_boundary(cbf, initial_state)
# print(boundary_pt)

# ------------------------------------------------- Define controller ------------------------------------------------------
sample_time = .001
controller = NominalQP(plant, clf, cbfs, alpha = 10.0, beta = 10.0, p = 1.0)

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
xlimits, ylimits = [-8, 8], [-8, 8]
plot_config = {
    "figsize": (5,5),
    "gridspec": (1,1,1),
    "widthratios": [1],
    "heightratios": [1],
    "axeslim": tuple(xlimits+ylimits),
    "path_length": 10,
    "numpoints": 1000,
    "drawlevel": True,
    "resolution": 50,
    "fps":120,
    "pad":2.0,
    "equilibria": False
}

logs = { "sample_time": sample_time, "P": P.tolist(), "Q": Q.tolist(), "clf_center": clf_center.tolist() }