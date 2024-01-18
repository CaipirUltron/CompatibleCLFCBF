import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

from common import rgb
from controllers.equilibrium_algorithms import check_equilibrium, compute_equilibria, generate_boundary, closest_compatible

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

# ----------------------------------------------------- Plotting ---------------------------------------------------------
fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Kernel-based CLF-CBF fitting")

sim.clf.plot_level(axes = ax, level = 23.0, axeslim = [-10, 10, -10, 10])
sim.cbf.plot_level(axes = ax, axeslim = [-10, 10, -10, 10])

# ----------------------------------------------------- Plotting ---------------------------------------------------------

# manual_mode = True
manual_mode = False
    
'''
Plots initial guesses
'''
if not manual_mode:
    num_pts = 10
    limits = [ [-10, 10], [-10, 10] ]
    # sols, log = generate_boundary(num_pts, cbf=sim.cbf, limits=limits)
    sols, log = generate_boundary(num_pts, plant=sim.plant, clf=sim.clf, cbf=sim.cbf, limits=limits, slack_gain=sim.p, clf_gain=sim.alpha)
    initial_guesses = []
    for sol in sols:
        initial_guesses.append(sol["x"])
        ax.plot( sol["x"][0], sol["x"][1], 'go', alpha=0.4 )

'''
Finds and plots the equilibrium points
'''
if manual_mode:
    is_equilibrium = False
    while not is_equilibrium:
        print("Didn't hit the equilibrium...")
        pt = plt.ginput(1)
        clicked_pt = np.array([ pt[0][0], pt[0][1] ])
        is_equilibrium, eq_pt = check_equilibrium(sim.plant, sim.clf, sim.cbf, clicked_pt, slack_gain=sim.p, clf_gain=sim.alpha)
    solutions = [eq_pt]
else:
    solutions, log = compute_equilibria(sim.plant, sim.clf, sim.cbf, initial_guesses, slack_gain=sim.p, clf_gain=sim.alpha)
    print("From " + str(log["num_trials"]) + " trials, algorithm converged " + str(log["num_success"]) + " times, and " + str(len(solutions)) + " solutions were found.")
    print("Algorithm efficiency = " + str( log["num_success"]/log["num_trials"] ))
    # P = closest_compatible(sim.plant, sim.clf, sim.cbf, solutions, slack_gain=sim.p, clf_gain=sim.alpha)
    # print(P)

num_sols = len(solutions)
for k in range(num_sols):
    sol = solutions[k]
    print("Solution " + str(k+1) + " = " + str(sol))
    ax.plot( sol["x"][0], sol["x"][1], 'ro' )

    '''
    Checks curvatures
    '''
    clf_curv = sim.clf.get_curvature(sol["x"])
    cbf_curv = sim.cbf.get_curvature(sol["x"])

    print("CLF curvature = " + str(clf_curv))
    print("CBF curvature = " + str(cbf_curv))
    print("Delta curvature = " + str(cbf_curv - clf_curv))

    if sol["stability"] > 0:
        print("Equilibrium point " + str(sol["x"]) + " is unstable, with value = " + str(sol["stability"]))
    else:
        print("Equilibrium point " + str(sol["x"]) + " is stable, with value = " + str(sol["stability"]))

plt.show()