import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

from common import rgb
from controllers.equilibrium_algorithms import check_equilibrium, compute_equilibria, closest_compatible, generate_boundary

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Kernel-based CLF-CBF fitting")

sim.clf.plot_level(axes = ax, level = 23.0, axeslim = [-10, 10, -10, 10])
sim.cbf.plot_level(axes = ax, axeslim = [-10, 10, -10, 10])

limits = [ [-10, 10], [-10, 10] ]
num_pts = 100
sols, log = generate_boundary(sim.cbf, num_pts, limits=limits)

print("From "+str(log["num_trials"])+" , algorithm converged "+str(log["num_success"])+" times.")
for sol in sols:
    x = sol["x"]
    ax.plot( x[0], x[1], 'og' )

plt.show()