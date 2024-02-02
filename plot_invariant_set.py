import sys, time
import importlib
import numpy as np
import matplotlib.pyplot as plt

from controllers.equilibrium_algorithms import compute_equilibria, plot_invariant, minimize_branch

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Invariant set plot for Kernel-based CLF-CBFs")
ax.set_aspect('equal', adjustable='box')

limits = [ [-6, 6],
           [-4, 8] ]

ax.set_xlim(limits[0][0], limits[0][1])
ax.set_ylim(limits[1][0], limits[1][1])

contour_boundary = sim.cbf.plot_levels(levels = [-0.4, -0.2, 0.0], ax=ax, limits=limits)
contour_invariant = plot_invariant(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, ax=ax, limits=limits, extended=False)

init_x_plot, = ax.plot([],[],'ob', alpha=0.5)
sol_x_plot, = ax.plot([],[],'or', alpha=0.8)
sol_x_minimize_plot, = ax.plot([],[],'og', alpha=0.8)

while True:
    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    init_x_plot.set_data([init_x[0]], [init_x[1]])

    # Find equilibrium point
    sol_eq = compute_equilibria(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, init_x=init_x, limits=limits)
    x_eq = sol_eq["x"]
    l_eq = sol_eq["lambda"]
    print(f"Equilibrium found point {x_eq}, with lambda = {l_eq}")
    sol_x_plot.set_data([x_eq[0]], [x_eq[1]])

    if "clf_contour" in locals():
        for coll in clf_contour.collections:
            coll.remove()
    V = sim.clf.function(x_eq)
    clf_contour = sim.clf.plot_levels(levels=[V], ax=ax, limits=limits)

    # Find minimum point in the invariant set branch
    sol_minimize = minimize_branch(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, init_x=init_x, limits=limits)
    x_min = sol_minimize["x"]
    l_min = sol_minimize["lambda"]
    print(f"Minimization found point {x_min}, with lambda = {l_min}")
    sol_x_minimize_plot.set_data([x_min[0]], [x_min[1]])

plt.show()