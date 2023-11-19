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
sim.plot_config["equilibria"] - False
plotSim = Plot2DSimulation( logs, sim.plant, sim.clf, sim.cbfs, plot_config = sim.plot_config )
plotSim.plot_frame(9.0)

pos = plt.ginput(1)
stability = compute_stability(sim.plant, sim.clf, sim.cbf, sim.eq_sol, c = sim.controller.p * sim.controller.alpha)

print("Stability = " + str(stability))

plt.show()