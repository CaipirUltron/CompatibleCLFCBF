import sys, time
import importlib
import numpy as np
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from common import rgb
from controllers.equilibrium_algorithms import check_equilibrium, compute_equilibria, closest_compatible

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Kernel-based CLF-CBF fitting")

# sim.clf.plot_level(axes = ax, level = 23.0, axeslim = [-10, 10, -10, 10])
# sim.cbf.plot_level(axes = ax, axeslim = [-10, 10, -10, 10])
# --------------------------------------------------------------------------------------------------------------

slack_gain = sim.p
clf_gain = sim.alpha

P = sim.clf.P
eigP = np.linalg.eigvals(P)
Q = sim.cbf.Q
p = sim.kernel.kernel_dim
rankQ = np.linalg.matrix_rank(Q)
dim_elliptical_manifold = rankQ - 1

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
        Jm = sim.kernel.jacobian(x)
        Null = sp.linalg.null_space(Jm.T)
        dimNull = Null.shape[1]

        def invariant(l, P, alpha):
            return (l/l0 * Q - P) @ m - Null @ alpha

        l_var = cp.Variable()
        alpha = cp.Variable(dimNull)
        P_var = cp.Variable((p,p), symmetric=True)

        objective = cp.Minimize( l_var )
        constraint = [ invariant(l_var, P_var, alpha) == 0
                      ,P_var >> P
                    #   ,m.T @ P_var @ m == 2*V
                      ,cp.lambda_max(P_var) <= np.max(eigP)
                       ]
        problem = cp.Problem(objective, constraint)

        try:
            problem.solve()
        except Exception as error:
            print("Problem cannot be computed. Error: " + str(error))

        print("Optimization status exit as \"" + str(problem.status) + "\".")

        if "optimal" in problem.status:
            print("Invariant error = " + str(np.linalg.norm( invariant(l_var.value, P_var.value, alpha.value) )) )
            print("Final level set V = " + str( 0.5 * m.T @ P_var.value @ m ) )
            print("Final P eigenvalues = " + str(np.linalg.eigvals(P_var.value)))
            print("Gradient norm = " + str(np.linalg.norm( P_var.value @ m )) )
            print("Minimum lambda = " + str(l_var.value))

        sim.clf.set_param(P=P_var.value)
        sim.clf.plot_level(axes = ax, level = 0.5 * m.T @ P_var.value @ m, axeslim = [-10, 10, -10, 10],color=mcolors.TABLEAU_COLORS['tab:blue'])
        sim.clf.set_param(P=P)

    else:
        print("No equilibrium point was found.")

plt.show()