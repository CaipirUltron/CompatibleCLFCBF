import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

from common import rgb
from controllers.equilibrium_algorithms import check_equilibrium, compute_equilibria2, closest_compatible

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Kernel-based CLF-CBF fitting")

sim.clf.plot_level(axes = ax, level = 23.0, axeslim = [-10, 10, -10, 10])
sim.cbf.plot_level(axes = ax, axeslim = [-10, 10, -10, 10])
# --------------------------------------------------------------------------------------------------------------

Q = sim.cbf.Q
p = sim.kernel.kernel_dim
rankQ = np.linalg.matrix_rank(Q)
dim_elliptical_manifold = rankQ - 1

limits = [ [-10, 10], [-10, 10] ]
init_x = [ 0.2 , 5.0 ]

pt = plt.ginput(1)
init_x = [ pt[0][0], pt[0][1] ]
init_theta = np.zeros(dim_elliptical_manifold).tolist()
init_alpha = np.zeros(p).tolist()

sol = compute_equilibria2(sim.plant, sim.clf, sim.cbf, init_x=init_x, init_theta=init_theta, init_alpha=init_alpha)

print("num trials = " + str(len(sol["init_x"])))

if sol["x"] != None:
    x = sol["x"]
    print("Solution found: " + str(sol))

    ax.plot(sol["init_x"][0],sol["init_x"][1],'ob',alpha=0.5)

    ax.plot(x[0],x[1],'or',alpha=0.8)

plt.show()