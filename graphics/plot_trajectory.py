import sys
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt
from graphics import Plot2DSimulation

plt.rcParams['text.usetex'] = True

simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples." + simulation_file, package=None)

try:
    with open(simulation_file + ".json") as file:
        print("Loading graphical simulation with " + simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate "+simulation_file + ".json")

plotSim = Plot2DSimulation(logs, sim.plant, sim.clf, sim.cbfs)
plotSim.plot_frame(5.0)

plt.savefig(simulation_file + ".eps", format='eps', transparent=True)
# plt.savefig(simulation_file + ".svg", format="svg",transparent=True)

plt.show()