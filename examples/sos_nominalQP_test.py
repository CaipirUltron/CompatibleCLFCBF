import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP
from common import create_quadratic, rot2D

# initial_state = [-4.2, 5.0]
initial_state = [-8.0, -2.15]
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

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
# Proot = 0.1*np.random.rand(p,p)
# P = Proot.T @ Proot

clf_eigs = np.array([5.0, 1.0])
clf_rotation = rot2D( np.deg2rad(-45) )
clf_center = np.array([4.0, -4.0])
P = create_quadratic(clf_eigs, clf_rotation, clf_center, p)

clf = KernelLyapunov(*initial_state, kernel=kernel, P=P)
clf.define_center( clf_center )

# ----------------------------------------------- Define CBF (sad smile) ---------------------------------------------------
# Qroot = 0.1*np.random.rand(p,p)
# Q = Qroot.T @ Qroot

cbf_eigs = np.array([1.0, 4.0])
cbf_rotation = rot2D( np.deg2rad(0) )
cbf_center = np.array([0.0, 0.0])
Q = create_quadratic(cbf_eigs, cbf_rotation, cbf_center, p)

cbf = KernelBarrier(*initial_state, kernel=kernel, Q=Q)

boundary_points = np.array([ [-4.0, 0.0], [-4.0, -1.0], [2.0, 0.5], [4.0, -1.0], [4.0, 0.0] ])
cbf.define_boundary( boundary_points )

cbfs = [cbf]

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