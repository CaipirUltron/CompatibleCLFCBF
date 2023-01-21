import sys
import importlib
import matplotlib.pyplot as plt
from graphics import SimulationMatplot

# Load simulation file
initialization_file = sys.argv[1]
sim = importlib.import_module("examples."+initialization_file.replace(".json",""), package=None)

print('Animating simulation...')
axes_lim = (-6,6,-6,6)

plotSim = SimulationMatplot(axes_lim, 50, initialization_file, sim.clf, sim.cbfs, draw_level=True)
plotSim.animate()
plt.show()