import sys, time
import importlib
import numpy as np
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from common import rgb
from controllers.equilibrium_algorithms import compute_equilibria, is_removable

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Kernel-based CLF-CBF fitting")

# --------------------------------------------------------------------------------------------------------------

slack_gain = sim.p
clf_gain = sim.alpha

P = sim.clf.P
eigP = np.linalg.eigvals(P)
Q = sim.cbf.Q
p = sim.kernel.kernel_dim
rankQ = np.linalg.matrix_rank(Q)
dim_elliptical_manifold = rankQ - 1
A_list = sim.kernel.get_A_matrices()

limits = [ [-10, 10], [-10, 10] ]

while True:
    
    sim.cbf.plot_level(axes = ax, axeslim = [-10, 10, -10, 10])
    pt = plt.ginput(1, timeout=0)
    ax.clear()
    init_x = [ pt[0][0], pt[0][1] ]

    t0 = time.time()
    sol = compute_equilibria(sim.plant, sim.clf, sim.cbf, {"slack_gain": sim.p, "clf_gain": sim.alpha}, init_x=init_x, limits=limits)
    delta_time = time.time() - t0
    print("Result took " +str(delta_time) + "s to compute.")
            
    if sol["x"] != None:

        x = sol["x"]
        l = sol["lambda"]
        V = sim.clf.function(x)
        l0 = slack_gain*clf_gain*V

        sim.clf.plot_level(axes = ax, level = V, axeslim = [-10, 10, -10, 10], color=mcolors.TABLEAU_COLORS['tab:cyan'])

        print("Initial level set V = " + str( V ) )
        print("Initial P eigenvalues = " + str(np.linalg.eigvals(P)))

        print("Solution found: " + str(sol))
        ax.plot(sol["init_x"][0],sol["init_x"][1],'ob',alpha=0.5)
        ax.plot(x[0],x[1],'or',alpha=0.8)

        m = sim.kernel.function(x)
        # Jm = sim.kernel.jacobian(x)
        # Null = sp.linalg.null_space(Jm.T)
        # dimNull = Null.shape[1]

        Pnew = is_removable(sol, sim.plant, sim.clf, sim.cbf)
        print("eigs of P = " + str( np.linalg.eigvals( Pnew ) ))

        sim.clf.set_param(P=Pnew)
        sim.clf.plot_level(axes = ax, level = 0.5 * m.T @ Pnew @ m, axeslim = [-10, 10, -10, 10],color=mcolors.TABLEAU_COLORS['tab:blue'])
        sim.clf.set_param(P=P)

    else:
        print("No equilibrium point was found.")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 6)

plt.show()