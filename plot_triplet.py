import sys, json, time
import importlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Invariant set plot for Kernel-based CLF-CBFs")
ax.set_aspect('equal', adjustable='box')

limits = sim.plot_config["limits"]
ax.set_xlim(limits[0][0], limits[0][1])
ax.set_ylim(limits[1][0], limits[1][1])

if hasattr(sim, "pts"):
    for pt in sim.pts:
        coords = np.array(pt["coords"])
        ax.plot(coords[0], coords[1], 'k*', alpha=0.6)

        if "gradient" in pt.keys():
            gradient_vec = coords + np.array(pt["gradient"])
            ax.plot([ coords[0], gradient_vec[0]], [ coords[1], gradient_vec[1]], 'k-', alpha=0.6)

contour_unsafe = sim.cbf.plot_levels(levels = [ -0.1*k for k in range(4,-1,-1) ], ax=ax, limits=limits)

sim.kerneltriplet.plot_invariant(ax)
sim.kerneltriplet.plot_attr(ax, "stable_equilibria", mcolors.BASE_COLORS["r"], 1.0)
sim.kerneltriplet.plot_attr(ax, "unstable_equilibria", mcolors.BASE_COLORS["g"], 0.8)

for k, seg in enumerate(sim.kerneltriplet.invariant_segs):
    message = f"Segment {k+1} is "
    if seg["removable"] == +1: message += "removable from the outside, "
    if seg["removable"] == -1: message += "removable from the inside, "
    if seg["removable"] == 0: message += "not removable, "
    critical = seg["segment_critical"]
    message += f"with critical value = {critical}."
    print(message)

init_x_plot, = ax.plot([],[],'ob', alpha=0.5)
while True:
    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    init_x_plot.set_data([init_x[0]], [init_x[1]])

    if "clf_contour" in locals():
        for coll in clf_contour:
            coll.remove()
        del clf_contour

    V = sim.clf.function(init_x)
    gradV = sim.clf.gradient(init_x)

    h = sim.cbf.function(init_x)
    gradh = sim.cbf.gradient(init_x)

    print(f"lambda = {sim.kerneltriplet.lambda_fun(init_x)}")

    print(f"V = {V}")
    print(f"||∇V|| = {np.linalg.norm(gradV)}")

    print(f"h = {h}")
    print(f"||∇h|| = {np.linalg.norm(gradh)}")

    clf_contour = sim.clf.plot_levels(levels=[V], ax=ax, limits=limits, spacing=0.1)

plt.show()