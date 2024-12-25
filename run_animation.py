import sys, json, time
import importlib
import matplotlib.pyplot as plt
from graphics import PlotQuadraticSim, PlotPFSimulation

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

try:
    with open("logs/"+simulation_file + ".json") as file:
        print("Loading graphical simulation with " + simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate " + simulation_file + ".json")

print('Animating simulation...')
time.sleep(1.5)

if not hasattr(sim, 'path'):
    plotSim = PlotQuadraticSim( logs, sim.plant, sim.clf, sim.cbfs, plot_config = sim.plot_config )
else:
    plotSim = PlotPFSimulation( sim.path, logs, sim.plant, sim.clf, sim.cbfs, plot_config = sim.plot_config )
    plotSim.main_ax.set_title("Path Following with Obstacle Avoidance using CLF-CBFs", fontsize=12)

figsize = (5, 5)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, layout="constrained")
fig.suptitle('CLF-CBF-QP Control Simulation', fontsize=12)
# fig.tight_layout(pad=1)
# ax.set_aspect('equal', adjustable='box')

plotSim.init_graphics(ax)

initial_time = 0.0
if len(sys.argv) > 2:
    initial_time = float(sys.argv[2])

animation = plotSim.animation(fig, initial_time)

# plotSim.animation.save(simulation_file + ".mp4", writer=anim.FFMpegWriter(fps=30, codec='h264'), dpi=100)

plt.show()