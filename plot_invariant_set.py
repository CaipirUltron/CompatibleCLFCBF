import sys, time
import importlib
import numpy as np
import matplotlib.pyplot as plt

from controllers.equilibrium_algorithms import compute_equilibria, plot_invariant

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Invariant set plot for Kernel-based CLF-CBFs")

limits = np.array([ [-10, 10], [-10, 10] ])

sim.cbf.plot_level(axes = ax, axeslim = limits.reshape(4,), level = -0.4)
sim.cbf.plot_level(axes = ax, axeslim = limits.reshape(4,), level = -0.2)
sim.cbf.plot_level(axes = ax, axeslim = limits.reshape(4,))
plot_invariant(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, ax=ax, limits=limits, extended=True)

init_x_plot, = ax.plot([],[],'ob', alpha=0.5)
sol_x_plot, = ax.plot([],[],'or', alpha=0.8)

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
        clf_contour = sim.clf.plot_level(axes = ax, level = V, axeslim = limits.reshape(4,))
    else:
        print("No equilibrium point was found.")

plt.show()