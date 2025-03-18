import sys, json, time
import importlib
import matplotlib.pyplot as plt
from graphics import PlotQuadraticSim, PlotPFSimulation

# Load simulation file
if len(sys.argv) < 2:
    raise Exception("Must specify: (i) simulation file and (ii) controller.")

# Load simulation
sim_config = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples.simulation."+sim_config, package=None)

# Get control mode
control_mode = sys.argv[2].replace(".json","")
simulation_file = sim_config+"_"+control_mode+".json"

try:
    with open("logs/"+simulation_file) as file:
        print("Loading graphical simulation with " + simulation_file)
        logs = json.load(file)
except IOError:
    print("Couldn't locate " + simulation_file)

print('Animating simulation...')
time.sleep(1.5)

if not hasattr(sim, 'path'):
    plotSim = PlotQuadraticSim( logs, sim.plant, sim.clf, sim.cbfs, plot_config = sim.plot_config )
else:
    plotSim = PlotPFSimulation( sim.path, logs, sim.plant, sim.clf, sim.cbfs, plot_config = sim.plot_config )
    # plotSim.main_ax.set_title("Path Following with Obstacle Avoidance using CLF-CBFs", fontsize=12)

figsize = (5, 5)

if sim.plant.n == 2:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, layout="constrained")
elif sim.plant.n == 3:
    fig = plt.figure(figsize=figsize)  # Square figure
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # ax.set_xticks([])
    # ax.set_yticks([])

    # fig.tight_layout()
else:
    raise Exception("Plotting not implemented")

# fig.suptitle('CLF-CBF-QP Control', fontsize=12)
# fig.suptitle('Undesirable Equilibrium Points in CLF-CBF QP Control', fontsize=12)

fig.tight_layout(pad=1)
ax.set_aspect('equal', adjustable='box')

plotSim.init_graphics(ax)