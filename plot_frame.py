import sys
import json
import importlib
import matplotlib.pyplot as plt
from graphics import Plot2DSimulation

# Load simulation file
if len(sys.argv) < 2:
    raise Exception("Must specify a simulation file and the desired time.")

simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

try:
    with open("logs/"+simulation_file + ".json") as file:
        print("Loading graphical simulation with "+simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate " + simulation_file + ".json")

time = float(sys.argv[2])
plotSim = Plot2DSimulation( logs, sim.plant, sim.clf, [sim.cbf], plot_config = sim.plot_config )
plotSim.plot_frame(time)
plt.show()