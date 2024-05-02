import numpy as np
import cvxpy as cp
import sys, importlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from common import symmetric_basis, create_quadratic, rot2D

np.set_printoptions(precision=4, suppress=True)

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ---------------------------------------- Plotting ----------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test CLF functions with lowerbound on max λ(Hv)")
ax.set_aspect('equal', adjustable='box')

limits = sim.limits
xmin, xmax, ymin, ymax = limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

kernel_dim = sim.kernel_dim
As = sim.kernel.Asum
As2 = As @ As

num_levels = 5
contour_unsafe = sim.cbf.plot_levels(ax=ax, levels = [ -(0.5/num_levels)*k for k in range(num_levels-1,-1,-1) ])

#--------------------------------------- Optimization probl. -------------------------------------

P_var = cp.Variable( (kernel_dim, kernel_dim), symmetric=True )
Pnom_var = cp.Parameter( (kernel_dim, kernel_dim), symmetric=True )
cost = cp.norm( P_var - Pnom_var )
constraints = [ P_var >> 0 ]
constraints += [ P_var @ col == 0 for col in As2.T if np.any(col != 0.0) ]
constraints += [ cp.lambda_max(P_var) <= 100.0 ]

prob = cp.Problem( cp.Minimize(cost), constraints )

def update_clf( Pnom , bypass=False):

    if not bypass:
        Pnom_var.value = Pnom
        prob.solve(solver="SCS", verbose=True, max_iters=10000)
        sim.kerneltriplet.clf.set_params(P=P_var.value)
        sim.kerneltriplet.P = P_var.value
    else:
        sim.kerneltriplet.clf.set_params(P=Pnom)
        sim.kerneltriplet.P = Pnom
    
    sim.kerneltriplet.clf.generate_contour()
    sim.kerneltriplet.update_invariant_set()

    return sim.kerneltriplet.P

while True:

    pt = plt.ginput(1, timeout=0)
    x = [ pt[0][0], pt[0][1] ]

    # P = np.random.randn(kernel_dim, kernel_dim)
    # Pnom_var.value = P.T @ P

    clf_eig = np.array([10, 1])
    clf_angle = -30
    clf_center = np.array([0, -3])
    Pnom = create_quadratic( eigen=clf_eig, R=rot2D(np.deg2rad(clf_angle)), center=clf_center, kernel_dim=kernel_dim )

    noise = np.zeros([kernel_dim, kernel_dim])
    for basis in symmetric_basis(kernel_dim): noise += 0.0*np.random.randn()*basis
    Pnom += noise.T @ noise

    # Pnom = np.zeros([kernel_dim, kernel_dim])
    # for basis in symmetric_basis(kernel_dim): Pnom += (np.random.randint(low=-50,high=50)*np.random.randn() + np.random.randint(low=-100,high=100))*basis

    P = update_clf(Pnom)
    print(f"P = \n {P}")
    print(f"λ(P) = {np.linalg.eigvals(P)}")

    print(f"Num of invariant lines = {len(sim.kerneltriplet.invariant_lines)}")

    sim.kerneltriplet.plot_invariant(ax)
    sim.kerneltriplet.plot_attr(ax, "stable_equilibria", mcolors.BASE_COLORS["r"], 1.0)
    sim.kerneltriplet.plot_attr(ax, "unstable_equilibria", mcolors.BASE_COLORS["g"], 0.8)

    # Plots resulting CLF contour
    if "clf_contour" in locals():
        for coll in clf_contour:
            coll.remove()
        del clf_contour

    # num_levels = 2
    # step = 100
    # levels = [ (k**3)*step for k in range(num_levels)]
    clf_contour = sim.kerneltriplet.clf.plot_levels(levels=[100], ax=ax, limits=limits, spacing=0.1)
    plt.pause(1e-4)