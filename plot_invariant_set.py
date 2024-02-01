import sys, time
import importlib
import numpy as np
import matplotlib.pyplot as plt

from controllers.equilibrium_algorithms import compute_equilibria, plot_invariant, q_function

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(figsize=(10, 5),constrained_layout=True)
ax1 = fig.add_subplot(121)
ax1.set_title("Invariant set plot for Kernel-based CLF-CBFs")
ax1.set_aspect('equal', adjustable='box')

ax2 = fig.add_subplot(122)
ax2.set_title("Q-function for Kernel-based CLF-CBFs")
# ax2.set_aspect('equal', adjustable='box')
# ax2.set_xlim(0, 1000)
# ax2.set_ylim(-1, 30)

limits = [ [-6, 6],
           [-4, 8] ]

# sim.cbf.plot_levels(levels = [-0.4, -0.2], ax=ax, limits=limits)
# contour_boundary = sim.cbf.plot_levels(levels = [-0.4, -0.2, 0.0], ax=ax, limits=limits)
contour_invariant = plot_invariant(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, ax=ax1, limits=limits, extended=True)

# qfun = q_function(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, ax=ax1, limits=limits, num_levels=20, max_level=1.0, extended=True, tol=1e-1)
# ax2.plot(qfun["lambdas"], qfun["levels"],'b.')

# for pt in qfun["points"]:
#     ax1.plot(pt[0],pt[1],'*k', alpha=0.5)

init_x_plot, = ax1.plot([],[],'ob', alpha=0.5)
sol_x_plot, = ax1.plot([],[],'or', alpha=0.8)

while True:
    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    init_x_plot.set_data([init_x[0]], [init_x[1]])

    t0 = time.time()
    sol = compute_equilibria(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, init_x=init_x, limits=limits)
    delta_time = time.time() - t0
    print("compute_equilibria() has returned in " +str(delta_time) + "s.")

    if sol["x"] != None:
        print("Solution found: " + str(sol))

        x = sol["x"]
        l = sol["lambda"]
        V = sim.clf.function(x)

        sol_x_plot.set_data([x[0]], [x[1]])

        if "clf_contour" in locals():
            for coll in clf_contour.collections:
                coll.remove()
        clf_contour = sim.clf.plot_levels(levels=[V], ax=ax1, limits=limits)
    else:
        print("No equilibrium point was found.")

plt.show()