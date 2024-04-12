import sys, importlib

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

limits = sim.limits
xmin, xmax, ymin, ymax = limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

if hasattr(sim, "boundary_pts"):
    for pt in sim.boundary_pts:
        coords = np.array(pt)
        ax.plot(coords[0], coords[1], 'k*', alpha=0.6)

if hasattr(sim, "skeleton_pts"):
    # for seg in sim.skeleton_pts:
        for pt in sim.skeleton_pts:
            ax.plot(pt[0], pt[1], 'b*', alpha=0.6)

if hasattr(sim, "quadratic_cbf"):
    sim.quadratic_cbf.plot_levels(ax=ax, levels = [0.0], color='g')

num_levels = 5
contour_unsafe = sim.cbf.plot_levels(ax=ax, levels = [ -(0.5/num_levels)*k for k in range(num_levels-1,-1,-1) ])

print(f"λ(lowerbound) = {np.linalg.eigvals(sim.kernel._reduced_lowerbound_matrix(sim.cbf.Q))}\n")

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

    clf_contour = sim.clf.plot_levels(ax=ax, levels=[V])

plt.show()