import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
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
ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

# if hasattr(sim, "pts"):
#     for pt in sim.pts:
#         coords = np.array(pt["coords"])
#         ax.plot(coords[0], coords[1], 'k*', alpha=0.6)

#         if "gradient" in pt.keys():
#             gradient_vec = coords + np.array(pt["gradient"])
#             ax.plot([ coords[0], gradient_vec[0]], [ coords[1], gradient_vec[1]], 'k-', alpha=0.6)

num_steps = 1000
time_text = ax.text(limits[0]-5.5, limits[1]+0.5, str("Step = 0"), fontsize=14)

def init():

    sim.kerneltriplet.update_invariant_set()

    for cbf_index, cbf in enumerate(sim.cbfs):
        cbf.plot_levels(levels = [ -0.1*k for k in range(4,-1,-1) ], ax=ax, limits=limits)
        sim.kerneltriplet.plot_invariant(ax, cbf_index)

    sim.kerneltriplet.plot_attr(ax, "boundary_equilibria", mcolors.BASE_COLORS["g"])
    sim.kerneltriplet.plot_attr(ax, "interior_equilibria", mcolors.BASE_COLORS["k"])

def update(i):
    if i <= num_steps:

        time_text.set_text("Step = " + str(i))

        deltaP = 0.02 * np.random.rand(sim.clf.kernel_dim, sim.clf.kernel_dim)
        sim.kerneltriplet.P += deltaP.T @ deltaP

        sim.kerneltriplet.update_invariant_set()

        for cbf_index, cbf in enumerate(sim.cbfs):
            sim.kerneltriplet.plot_invariant(ax, cbf_index)

        sim.kerneltriplet.plot_attr(ax, "boundary_equilibria", mcolors.BASE_COLORS["g"])
        sim.kerneltriplet.plot_attr(ax, "interior_equilibria", mcolors.BASE_COLORS["k"])

fps = 60
animation = anim.FuncAnimation(fig, func=update, init_func=init, interval=1000/fps, repeat=False, cache_frame_data=False)
plt.show()

# Pnew = sim.kerneltriplet.compatibilize()