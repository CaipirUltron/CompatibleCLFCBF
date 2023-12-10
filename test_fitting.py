import numpy as np
import matplotlib.pyplot as plt

from common import rgb
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import compute_equilibria
from dynamic_systems import ConservativeAffineSystem

initial_state = [3.2, 3.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
max_degree = 2
kernel = Kernel(*initial_state, degree = max_degree)
kern_dim = kernel.kernel_dim
print(kernel)
print("Kernel dimension = " + str(kernel.kernel_dim))

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
base_level = 16.0
points = []
points += [{ "point": [ 0.0,  0.0], "level": 0.0 }]
points += [{ "point": [ 3.0,  3.0], "level": base_level, "gradient": [ 1.0,  1.0] }]
points += [{ "point": [-3.0,  3.0], "level": base_level, "gradient": [-1.0,  1.0] }]
points += [{ "point": [ 0.0,  5.0],                      "gradient": [ 0.0,  1.0], "curvature": -1.6 }]

clf = KernelLyapunov(*initial_state, kernel=kernel, points=points)

# ----------------------------------------------------- Define CBF ---------------------------------------------------------
points = []
points += [{ "point": [ 2.0,  2.0], "level": 0.0, "gradient": [ 1.0,  1.0] }]
points += [{ "point": [ 2.0, -2.0], "level": 0.0, "gradient": [ 1.0, -1.0] }]
points += [{ "point": [-2.0, -2.0], "level": 0.0, "gradient": [-1.0, -1.0] }]
points += [{ "point": [-2.0,  2.0], "level": 0.0, "gradient": [-1.0,  1.0] }]
points += [{ "point": [ 0.0,  2.0], "level": 0.0, "gradient": [ 0.0,  1.0] }]
points += [{ "point": [ 0.0, -2.0], "level": 0.0, "gradient": [ 0.0, -1.0] } ]
points += [{ "point": [ 2.0,  0.0], "level": 0.0 }]
points += [{ "point": [-2.0,  0.0], "level": 0.0 }]

displacement = 3*np.array([0,1])
for pt in points:
    pt["point"] = ( np.array(pt["point"]) + displacement ).tolist()

cbf = KernelBarrier(*initial_state, kernel=kernel, points=points)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Kernel-based CLF-CBF fitting")

clf.plot_level(axes = ax, level = 26.1, axeslim = [-10, 10, -10, 10])
cbf.plot_level(axes = ax, axeslim = [-10, 10, -10, 10])

arrow_width = 0.005
for pt in clf.point_list:
    ax.plot(pt["point"][0], pt["point"][1], 'o', color="blue")
    if "gradient" in pt.keys():
        if "curvature" in pt.keys():
            color = rgb(-10.0, 10.0, pt["curvature"])
            ax.quiver(pt["point"][0], pt["point"][1], pt["gradient"][0], pt["gradient"][1], color=color, width=arrow_width)
        else:
            ax.quiver(pt["point"][0], pt["point"][1], pt["gradient"][0], pt["gradient"][1], width=arrow_width)

for pt in cbf.point_list:
    ax.plot(pt["point"][0], pt["point"][1], 'o', color="green")
    if "gradient" in pt.keys():
        if "curvature" in pt.keys():
            color = rgb(-10.0, 10.0, pt["curvature"])
            ax.quiver(pt["point"][0], pt["point"][1], pt["gradient"][0], pt["gradient"][1], color=color, width=arrow_width)
        else:
            ax.quiver(pt["point"][0], pt["point"][1], pt["gradient"][0], pt["gradient"][1], width=arrow_width)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
max_degree = 2
kernel = Kernel(*initial_state, degree = max_degree)
kern_dim = kernel.kernel_dim
print(kernel)
print("Dimension of kappa space = " + str(len(kernel.get_N_matrices())))

# -------------------------------------------------- Define system ---------------------------------------------------------
F = np.zeros([kern_dim,kern_dim])
def g(state):
    return np.eye(m)
plant = ConservativeAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

# --------------------------------------------------- Verifying equilibrium ---------------------------------------------------------
eq_pt = [0, 5]

solutions, log = compute_equilibria(plant, clf, cbf, [eq_pt], slack_gain=1, clf_gain=1)

for sol in solutions:
    print("Solution was found = " + str(sol))
    clf_curv = clf.get_curvature(sol["x"])
    cbf_curv = cbf.get_curvature(sol["x"])

print("Curvature of V at eq. = " + str( clf_curv ))
print("Curvature of h at eq. = " + str( cbf_curv ))
print("Diff between curvatures at eq. = " + str( cbf_curv - clf_curv ))


plt.show()