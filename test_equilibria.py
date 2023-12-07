import sys
import json
import importlib
import numpy as np
import matplotlib.pyplot as plt

from graphics import Plot2DSimulation
from controllers.equilibrium_algorithms import compute_equilibria_algorithm9

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

try:
    with open("logs/"+simulation_file + ".json") as file:
        print("Loading graphical simulation with "+simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate " + simulation_file + ".json")

# -----------------------------------Plots starting of simulation ------------------------------------------
plotSim = Plot2DSimulation( logs, sim.plant, sim.clf, sim.cbfs, plot_config = sim.plot_config )
plotSim.plot_frame(3.0)

# manual_mode = True
manual_mode = False

'''
Manually selects initial guesses
'''
if manual_mode:
    pt = plt.ginput(1)
    initial_guesses = [[ pt[0][0], pt[0][1] ]]

'''
Plots initial guesses
'''
if not manual_mode:
    res_x, res_y = 10, 10       # Density of points per axis
    min_x, min_y = -8, -8     # lower limits for point generation 
    max_x, max_y = 8, 8       # upper limits for point generation
    initial_guesses = []
    xv, yv = np.meshgrid(np.linspace(min_x, max_x, res_x), np.linspace(min_y, max_y, res_y), indexing='xy')
    for i in range(res_x):
        for j in range(res_y):
            pt = [ xv[i,j], yv[i,j] ]
            initial_guesses.append(pt)
            plotSim.main_ax.plot( pt[0], pt[1], 'g.', alpha=0.3 )

'''
Finds and plots the equilibrium points
'''
solutions, log = compute_equilibria_algorithm9(sim.plant, sim.clf, sim.cbf, initial_guesses, c = 1)
num_sols = len(solutions)
print("From " + str(log["num_trials"]) + " trials, algorithm converged " + str(log["num_success"]) + " times, and " + str(num_sols) + " solutions were found.")
print("Algorithm efficiency = " + str( log["num_success"]/log["num_trials"] ))
for k in range(num_sols):
    sol = solutions[k]
    print("Solution " + str(k+1) + " = " + str(sol))
    plotSim.main_ax.plot( sol["x"][0], sol["x"][1], 'ro' )

    if sol["stability"] > 0:
        print("Equilibrium point " + str(sol["x"]) + " is unstable, with value = " + str(sol["stability"]))
    else:
        print("Equilibrium point " + str(sol["x"]) + " is stable, with value = " + str(sol["stability"]))

plt.show()