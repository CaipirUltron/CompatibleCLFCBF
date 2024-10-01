import sys, importlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

''' ----------------------------------- Initialize plot ----------------------------------- '''

fig = plt.figure(constrained_layout=True)

ax = fig.add_subplot(111)
ax.set_title("Interactive Plot for Polynomial CLF-CBFs")
ax.set_aspect('equal', adjustable='box')

limits = sim.limits
xmin, xmax, ymin, ymax = limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

font = {'weight': 'bold', 'size': 8}
plt.rc('font', **font)

''' -------------------- Plot CBF countours and corresponding invariant set ----------------------- '''

num_lvls, final_lvl = 10, -0.4
for cbf_index, cbf in enumerate(sim.cbfs):
    lvls = [ final_lvl*(k/(num_lvls-1)) for k in range(num_lvls) ]
    contour_unsafe = cbf.plot_levels(ax=ax, levels = lvls + [0.0] )
    center = cbf.find_center()
    ax.text(center[0], center[1], f"CBF{cbf_index+1}")

''' ------------------------------ Plot desired invariant set ------------------------------------- '''

invariant_set_to_plot = 1
sim.kerneltriplet.plot_invariant(ax, invariant_set_to_plot-1)

''' ---------------------------- Plot found equilibrium points  ----------------------------------- '''

sim.kerneltriplet.plot_attr(ax, "stable_equilibria", mcolors.BASE_COLORS["r"], alpha=0.8)
sim.kerneltriplet.plot_attr(ax, "unstable_equilibria", mcolors.BASE_COLORS["g"], alpha=0.8)

''' ---------------------------------- Interactive plot ------------------------------------------- '''

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

    print(f"V = {V}")
    print(f"||∇V|| = {np.linalg.norm(gradV)}")

    for cbf_index, cbf in enumerate(sim.cbfs):
        h = cbf.function(init_x)
        gradh = cbf.gradient(init_x)

        print(f"CBF {cbf_index+1} value at this point = {h}")
        print(f"CBF {cbf_index+1} λ at this point = {sim.kerneltriplet.lambda_fun(init_x, cbf_index)}")
        print(f"CBF {cbf_index+1} ||∇h|| at this point = {np.linalg.norm(gradh)}")

    clf_contour = sim.clf.plot_levels(ax=ax, levels=[V])
    plt.pause(0.001)