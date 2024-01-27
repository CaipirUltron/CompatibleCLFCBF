import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP
from common import create_quadratic, rot2D

initial_state = [0.2, 5.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(*initial_state, degree=1)
kernel_dim = kernel.kernel_dim
print(kernel)

# -------------------------------------------------- Define system ---------------------------------------------------------
fx, fy = 0.0, 0.0                       # constant force with fx, fy components
F = np.zeros([kernel_dim,kernel_dim])
F[1,0], F[2,0] = fx, fy

def g(state):
    return np.eye(m)
plant = ConservativeAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

# --------------------------------------------- Define CLF (quadratic) -----------------------------------------------------
# clf_eig = [ 1.0, 0.25 ]
clf_eig = 0.03*np.array([ 10.0, 1.0 ])
clf_angle = np.pi/5
clf_center = [0.0, 0.0]
clf = KernelLyapunov(*initial_state, kernel=kernel, P=create_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim))

# --------------------------------------------- Define CBF (quadratic) -----------------------------------------------------
cbf_eig = [ 0.2, 1.2 ]
cbf_angle = 0.0
cbf_center = [0.0, 3.0]
cbf = KernelBarrier(*initial_state, kernel=kernel, Q=create_quadratic(eigen=cbf_eig, R=rot2D(cbf_angle), center=cbf_center, kernel_dim=kernel_dim))

# ------------------------------------------------- Define controller ------------------------------------------------------
T = 15
sample_time = .002
p, alpha, beta = 1.0, 1.0, 1.0
controller = NominalQP(plant, clf, cbf, alpha, beta, p, dt=sample_time)

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
xlimits, ylimits = [-8, 8], [-8, 8]
plot_config = {
    "figsize": (5,5), "gridspec": (1,1,1), "widthratios": [1], "heightratios": [1], "axeslim": tuple(xlimits+ylimits),
    "path_length": 10, "numpoints": 1000, "drawlevel": True, "resolution": 50, "fps":30, "pad":2.0, "invariants": True, "equilibria": True, "arrows": True
}

logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": cbf.Q.tolist() }