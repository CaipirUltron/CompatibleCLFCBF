import sys
import importlib

import numpy as np
import matplotlib.pyplot as plt

from controllers.equilibrium_algorithms import compute_equilibria, plot_invariant, optimize_branch

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Invariant set plot for Kernel-based CLF-CBFs")
ax.set_aspect('equal', adjustable='box')

limits = (9*np.array([[-1, 1],[-1, 1]])).tolist()

ax.set_xlim(limits[0][0], limits[0][1])
ax.set_ylim(limits[1][0], limits[1][1])

contour_boundary = sim.cbf.plot_levels(levels = [ -0.1*k for k in range(4,-1,-1) ], ax=ax, limits=limits)
contour_invariant = plot_invariant(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, ax=ax, limits=limits, extended=False)

init_x_plot, = ax.plot([],[],'ob', alpha=0.5)
sol_x_plot, = ax.plot([],[],'ok', alpha=0.8)
sol_x_minimize_plot, = ax.plot([],[],'og', alpha=0.8)
sol_x_maximize_plot, = ax.plot([],[],'or', alpha=0.8)

while True:
    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    init_x_plot.set_data([init_x[0]], [init_x[1]])

    if "clf_contour" in locals():
        for coll in clf_contour.collections:
            coll.remove()
        del clf_contour

    # Find equilibrium point
    sol_eq = compute_equilibria(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, init_x=init_x, limits=limits)
    if sol_eq["x"] != None:
        x_eq = sol_eq["x"]
        z_eq = sim.kernel.function(x_eq)
        l_eq = sol_eq["lambda"]
        type = sol_eq["type"]
        print(f"{type} equilibrium point found at {x_eq}, with lambda = {l_eq}")
        sol_x_plot.set_data([x_eq[0]], [x_eq[1]])

        # Find minimum point in the invariant set branch
        # min_sol, max_sol = optimize_branch(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, init_pt=x_eq, init_lambda=l_eq)
        # print(f"Min = {min_sol}")
        # print(f"Max = {max_sol}")

        # x_min = min_sol["pt"]
        # x_max = max_sol["pt"]

        # if x_min != None:
        #     sol_x_minimize_plot.set_data([x_min[0]], [x_min[1]])
        # if x_max != None:
        #     sol_x_maximize_plot.set_data([x_max[0]], [x_max[1]])

    else:
        print("No equilibrium point was found.")
        sol_x_plot.set_data([],[])
        sol_x_minimize_plot.set_data([],[])
        sol_x_maximize_plot.set_data([],[])

    V = sim.clf.function(init_x)
    print(f"V = {V}")
    clf_contour = sim.clf.plot_levels(levels=[V], ax=ax, limits=limits)

plt.show()