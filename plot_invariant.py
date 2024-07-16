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

# if hasattr(sim, "boundary_pts"):
#     for pt in sim.boundary_pts:
#         coords = np.array(pt)
#         ax.plot(coords[0], coords[1], 'k*', alpha=0.6)

# if hasattr(sim, "skeleton_pts"):
#     # for seg in sim.skeleton_pts:
#         for pt in sim.skeleton_pts:
#             ax.plot(pt[0], pt[1], 'b*', alpha=0.6)

# if hasattr(sim, "quadratic_cbf"):
#     sim.quadratic_cbf.plot_levels(ax=ax, levels = [0.0], color='g')

# print(f"λ(M(P)) = {np.linalg.eigvals(sim.kernel.get_lowerbound(sim.clf.P))}\n")
# print(f"λ(M(Q)) = {np.linalg.eigvals(sim.kernel.get_lowerbound(sim.cbf.Q))}\n")

num_levels = 5
for k in range(len(sim.cbfs)):
    contour_unsafe = sim.cbfs[k].plot_levels(ax=ax, levels = [ -(0.5/num_levels)*k for k in range(num_levels-1,-1,-1) ])
    sim.kerneltriplet.plot_invariant(ax, k)

sim.kerneltriplet.plot_attr(ax, "stable_equilibria", mcolors.BASE_COLORS["r"], 1.0)
sim.kerneltriplet.plot_attr(ax, "unstable_equilibria", mcolors.BASE_COLORS["g"], 0.8)

for cbf_index in range(len(sim.cbfs)):
    for k, seg in enumerate(sim.kerneltriplet.invariant_segs[cbf_index]):
        message = f"Segment {k+1} of CBF {cbf_index+1} is "
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

    print(f"V = {V}")
    print(f"||∇V|| = {np.linalg.norm(gradV)}")

    for cbf_index, cbf in enumerate(sim.cbfs):
        h = cbf.function(init_x)
        gradh = cbf.gradient(init_x)

        print(f"CBF {cbf_index+1} value at this point = {h}")
        print(f"CBF {cbf_index+1} λ at this point = {sim.kerneltriplet.lambda_fun(init_x, cbf_index)}")
        print(f"CBF {cbf_index+1} ||∇h|| at this point = {np.linalg.norm(gradh)}")

        # sim.kerneltriplet.plot_removable_areas(ax, cbf_index)

    clf_contour = sim.clf.plot_levels(ax=ax, levels=[V])

plt.show()