import sys
import json
import importlib
import matplotlib.pyplot as plt
from graphics import PlotQuadraticSim

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

figsize= (5, 5)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, layout="constrained")
fig.suptitle('CLF-CBF-QP Controller')

plotSim = PlotQuadraticSim( logs, sim.plant, sim.clf, sim.cbfs, plot_config = sim.plot_config )

plotSim.init_graphics(ax)

t = float(sys.argv[2])
plotSim.plot_frame(t)

plt.show()