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

sim.clf.plot_level(axes = ax, level = 23.154, axeslim = [-10, 10, -10, 10])
sim.cbf.plot_level(axes = ax, axeslim = [-10, 10, -10, 10])

limits = [ [-10, 10], [5, 10] ]
num_pts = 15
# sols, log = generate_boundary(num_pts, plant=sim.plant, clf=sim.clf, cbf=sim.cbf, limits=limits, slack_gain=sim.p, clf_gain=sim.alpha)
sols, log = generate_boundary(num_pts, cbf=sim.cbf, limits=limits)

initial_guesses = []
for sol in sols:
    x = sol["x"]
    initial_guesses.append(x)
    ax.plot( x[0], x[1], 'og', alpha=0.3 )

eq_sols, log = compute_equilibria(sim.plant, sim.clf, sim.cbf, initial_guesses, slack_gain=sim.p, clf_gain=sim.alpha)

if len(eq_sols) > 0:
    Pnew = closest_compatible(sim.plant, sim.clf, sim.cbf, eq_sols, slack_gain=sim.p, clf_gain=sim.alpha, c_lim=5.0)
    sim.clf.set_param(P=Pnew)
    sim.clf.plot_level(axes = ax, level = 23.154, axeslim = [-10, 10, -10, 10])
    print("CLF is compatible now.")

eq_sols, log = compute_equilibria(sim.plant, sim.clf, sim.cbf, initial_guesses, slack_gain=sim.p, clf_gain=sim.alpha)
for sol in eq_sols:
    ax.plot( sol["x"][0], sol["x"][1], 'ro' )

plt.show()