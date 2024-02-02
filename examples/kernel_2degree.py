import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP

initial_state = [0.5, 6.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(*initial_state, degree=2)
kernel_dim = kernel.kernel_dim
print(kernel)

# -------------------------------------------------- Define system ---------------------------------------------------------
fx, fy = 0.0, 0.0                       # constant force with fx, fy components
F = np.zeros([kernel_dim,kernel_dim])
F[1,0], F[2,0] = fx, fy

def g(state):
    return np.eye(m)
plant = ConservativeAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
base_level = 16.0
points = []
points += [{ "point": [ 0.0,  -2.0], "level": 0.0 }]
points += [{ "point": [ 3.0,  3.0], "level": base_level, "gradient": [ 1.0,  1.0] }]
points += [{ "point": [-3.0,  3.0], "level": base_level, "gradient": [-1.0,  1.0] }]
points += [{ "point": [ 0.0,  5.0],                      "gradient": [ 0.0,  1.0], "curvature": -0.6 }]
# points += [{ "point": [ 0.0,  5.0],                      "gradient": [ 1.8,  1.0] }]
clf = KernelLyapunov(*initial_state, kernel=kernel, points=points)

# clf_eig = 0.05*np.array([ 12.0, 1.0 ])
# clf_angle = np.pi/100
# clf_center = [0.0, -2.0]
# clf = KernelLyapunov(*initial_state, kernel=kernel, P=create_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim))

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
T = 15
sample_time = .002
p, alpha, beta = 1.0, 1.0, 1.0
controller = NominalQP(plant, clf, cbf, alpha, beta, p, dt=sample_time)

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
xlimits, ylimits = [-6, 6], [-4, 8]
plot_config = {
    "figsize": (5,5), "gridspec": (1,1,1), "widthratios": [1], "heightratios": [1], "limits": [xlimits, ylimits],
    "path_length": 10, "numpoints": 1000, "drawlevel": True, "resolution": 50, "fps":30, "pad":2.0, "invariants": True, "equilibria": True, "arrows": True
}

logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": cbf.Q.tolist() }