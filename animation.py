import sys
import json
import importlib
import matplotlib.pyplot as plt
from graphics import SimulationMatplot

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

try:
    with open(simulation_file + ".json") as file:
        print("Loading graphical simulation with "+simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate "+simulation_file + ".json")

print('Animating simulation...')
plotSim = SimulationMatplot( logs, sim.clf, sim.cbfs )
plotSim.animate()

plt.show()