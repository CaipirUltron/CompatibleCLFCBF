import sys
import json
import importlib
import numpy as np
import matplotlib.pyplot as plt
from graphics import Plot2DSimulation
from controllers import compute_equilibria_algorithm7, find_nearest_boundary, find_nearest_det, compute_stability

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
plotSim.plot_frame(9.0)

print("The following equilibrium points were found:")
for eq in logs["equilibria"]:
    print(str(eq["point"]) + ", with stability value = " + str(eq["stability"]))
    plotSim.main_ax.plot( eq["point"][0], eq["point"][1], 'ro' )

# initial_guess = sim.initial_state
# plotSim.main_ax.plot( initial_guess[0], initial_guess[1], 'g*' )

# solution = compute_equilibria_algorithm7(sim.plant, sim.clf, sim.cbf, initial_guess, c = 1)
# plotSim.main_ax.plot( solution["boundary_start"][0], solution["boundary_start"][1], 'ko' )

# if solution != None:
#     print(solution)
#     print(str(solution["message"]))
#     plotSim.main_ax.plot( solution["point"][0], solution["point"][1], 'ro' )

plt.show()