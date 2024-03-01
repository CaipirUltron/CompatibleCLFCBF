import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# from controllers.equilibrium_algorithms import compute_equilibria, optimize_branch

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

num_steps = 1000
time_text = ax.text(0.5, 0.5, str("Time = "), fontsize=14)

def init():

    contour_unsafe = sim.cbf.plot_levels(levels = [ -0.1*k for k in range(4,-1,-1) ], ax=ax, limits=limits)
    graphical_elements = contour_unsafe

    return graphical_elements

def update(i):

    if i <= num_steps:
        
        if "inv_contour" in locals():
            for coll in inv_contour:
                coll.remove()
            del inv_contour

        time_text.set_text("Time = " + '{:^2f}'.format(i*1/fps) + "s")

        deltaP = np.random.rand(sim.clf.kernel_dim, sim.clf.kernel_dim)
        Pnew = sim.clf.P + deltaP.T @ deltaP
        sim.clf.set_param(P=Pnew)

        sim.kerneltriplet.invariant_set()
        inv_contour = sim.kerneltriplet.plot_invariant(ax=ax, extended=False)
    
    graphical_elements = []
    # graphical_elements += inv_contour
    graphical_elements.append(time_text)

    return graphical_elements

fps = 60
animation = anim.FuncAnimation(fig, func=update, init_func=init, interval=1000/fps, repeat=False, cache_frame_data=False)
plt.show()