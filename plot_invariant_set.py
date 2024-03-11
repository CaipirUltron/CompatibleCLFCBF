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
sim.kerneltriplet.plot_attr(ax, "boundary_equilibria", mcolors.BASE_COLORS["g"])
sim.kerneltriplet.plot_attr(ax, "interior_equilibria", mcolors.BASE_COLORS["k"])

init_x_plot, = ax.plot([],[],'ob', alpha=0.5)
while True:
    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    init_x_plot.set_data([init_x[0]], [init_x[1]])

    if "clf_contour" in locals():
        for coll in clf_contour:
            coll.remove()
        del clf_contour

    print(f"lambda = {sim.kerneltriplet.compute_lambda(init_x)}")

    V = sim.clf.function(init_x)
    clf_contour = sim.clf.plot_levels(levels=[V], ax=ax, limits=limits, spacing=0.5)

plt.show()