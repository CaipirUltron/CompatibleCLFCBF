import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP
from common import create_quadratic, rot2D

initial_state = [+3.2, 3.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
max_degree = 2
kernel = Kernel(*initial_state, degree = max_degree)
kern_dim = kernel.kernel_dim
print(kernel)

# -------------------------------------------------- Define system ---------------------------------------------------------
F = np.zeros([kern_dim,kern_dim])
def g(state):
    return np.eye(m)
plant = ConservativeAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
points_dict = { 2.0: [ [0.0, 2.0], [4.0, 1.0], [2.0, -2.0], [-2.0, -2.0], [-4.0, 1.0], [0.0, -4.0] ], 
                0.0: [ [0.0, -2.0] ] }
clf = KernelLyapunov(*initial_state, kernel=kernel, points = points_dict)

# ----------------------------------------------------- Define CBF ---------------------------------------------------------
boundary_points = [ [-4.0, 0.0], [-4.0, -1.0], [-2.0, 0.5], [2.0, 0.5], [4.0, -1.0], [4.0, 0.0], [0.0, 1.0], [0.0, -0.5] ]   # (sad   smile)
# boundary_points = [ [-4.0, 0.0], [-4.0, 1.0], [-2.0, -0.5], [2.0, -0.5], [4.0, 1.0], [4.0, 0.0] ] # (happy smile)

cbf = KernelBarrier(*initial_state, kernel=kernel, boundary_points=boundary_points)
cbfs = [cbf]

# ------------------------------------------------- Define controller ------------------------------------------------------
sample_time = .002
controller = NominalQP(plant, clf, cbfs, alpha = 10.0, beta = 10.0, p = 1.0, dt=sample_time)

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
    "equilibria": True
}

logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": cbf.Q.tolist() }