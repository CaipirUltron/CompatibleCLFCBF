import sys
import json
import importlib
import matplotlib.pyplot as plt

from common import Rect
from dynamic_systems import Unicycle
from graphics import Plot2DSimulation, PlotUnicycleSimulation

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

try:
    with open(simulation_file + ".json") as file:
        print("Loading graphical simulation with "+simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate " + simulation_file + ".json")

print('Animating simulation...')
plot_config = {
    "figsize": (5,5),
    "gridspec": (1,1,1),
    "widthratios": [1],
    "heightratios": [1],
    "axeslim": (-6,6,-6,6),
    "drawlevel": True,
    "resolution": 50,
    "fps":50,
    "pad":2.0
}

if type(sim.plant) == Unicycle:
    plot_config["radius"] = sim.controller.radius
    plotSim = PlotUnicycleSimulation( logs, sim.plant, sim.clf, sim.cbfs, plot_config = plot_config )
else:
    plotSim = Plot2DSimulation( logs, sim.plant, sim.clf, sim.cbfs, plot_config = plot_config )

plotSim.animate()
# plotSim.animation.save(simulation_file + ".mp4", writer=anim.FFMpegWriter(fps=30, codec='h264'), dpi=100)
plt.show()