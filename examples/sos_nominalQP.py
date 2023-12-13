import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP
from common import create_quadratic, rot2D

initial_state = [5.2, 3.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
max_degree = 2
kernel = Kernel(*initial_state, degree = max_degree)
kern_dim = kernel.kernel_dim
print(kernel)
print("Dimension of kappa space = " + str(len(kernel.get_N_matrices())))

# -------------------------------------------------- Define system ---------------------------------------------------------
fx, fy = 10.0, -10.0                       # constant force with fx, fy components
F = np.zeros([kern_dim,kern_dim])
F[1,0], F[2,0] = fx, fy

def g(state):
    return np.eye(m)
plant = ConservativeAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
base_level = 16.0
points = []
points += [{ "point": [ 0.0,  0.0], "level": 0.0 }]
points += [{ "point": [ 3.0,  3.0], "level": base_level, "gradient": [ 1.0,  1.0] }]
points += [{ "point": [-3.0,  3.0], "level": base_level, "gradient": [-1.0,  1.0] }]
points += [{ "point": [ 0.0,  5.0],                      "gradient": [ 0.0,  1.0], "curvature": -0.2 }]

clf = KernelLyapunov(*initial_state, kernel=kernel, points=points)

# eig = [1.0, 1.0]
# center = [0.0, 0.0]
# P = create_quadratic(eig, rot2D(0.0), center, kern_dim)
# clf = KernelLyapunov(*initial_state, kernel=kernel, P=P)

# ----------------------------------------------------- Define CBF ---------------------------------------------------------
# boundary_points = [ [-4.0, 0.0], [-4.0, -1.0], [-2.0, 0.5], [2.0, 0.5], [4.0, -1.0], [4.0, 0.0], [0.0, 1.0], [0.0, -0.5] ]   # (sad smile)
# boundary_points = [ [-4.0, 0.0], [-4.0, 1.0], [-2.0, -0.5], [2.0, -0.5], [4.0, 1.0], [4.0, 0.0] ]                            # (bell shaped)

points = []
points += [{ "point": [ 2.0,  2.0], "level": 0.0, "gradient": [ 1.0,  1.0] }]
points += [{ "point": [ 2.0, -2.0], "level": 0.0, "gradient": [ 1.0, -1.0] }]
points += [{ "point": [-2.0, -2.0], "level": 0.0, "gradient": [-1.0, -1.0] }]
points += [{ "point": [-2.0,  2.0], "level": 0.0, "gradient": [-1.0,  1.0] }]
points += [{ "point": [ 0.0,  2.0], "level": 0.0, "gradient": [ 0.0,  1.0] }]
points += [{ "point": [ 0.0, -2.0], "level": 0.0, "gradient": [ 0.0, -1.0] }]
points += [{ "point": [ 2.0,  0.0], "level": 0.0 }]
points += [{ "point": [-2.0,  0.0], "level": 0.0 }]

displacement = 3*np.array([0,1])
for pt in points:
    pt["point"] = ( np.array(pt["point"]) + displacement ).tolist()

cbf = KernelBarrier(*initial_state, kernel=kernel, points=points)
# cbf = KernelBarrier(*initial_state, kernel=kernel, boundary_points=boundary_points)

# ------------------------------------------------- Define controller ------------------------------------------------------
T = 30
sample_time = .002
p, alpha, beta = 1.0, 1.0, 1.0
controller = NominalQP(plant, clf, cbf, alpha, beta, p, dt=sample_time)

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
xlimits, ylimits = [-8, 8], [-8, 8]
plot_config = {
    "figsize": (5,5), "gridspec": (1,1,1), "widthratios": [1], "heightratios": [1], "axeslim": tuple(xlimits+ylimits),
    "path_length": 10, "numpoints": 1000, "drawlevel": True, "resolution": 50, "fps":30, "pad":2.0, "equilibria": True
}

logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": cbf.Q.tolist() }