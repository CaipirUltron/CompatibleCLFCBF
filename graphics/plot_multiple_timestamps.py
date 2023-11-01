import sys
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt
from graphics import Plot2DSimulation, PlotPFSimulation

plt.rcParams['text.usetex'] = True

simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples." + simulation_file, package=None)

try:
    with open("logs/" + simulation_file + ".json") as file:
        print("Loading graphical simulation with " + simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate "+simulation_file + ".json")

font_size = 14
configuration = {
    "figsize": (8,4),
    "gridspec": (1,2,1),
    "widthratios": [1, 1],
    "heightratios": [1],
    "axeslim": (-6,6,-6,6),
    "path_length": 10,
    "numpoints": 1000,
    "drawlevel": True,
    "resolution": 50,
    "fps":50,
    "pad":2.0,
    "equilibria": True
}

if hasattr(sim, 'path'):
    plotSim = PlotPFSimulation( sim.path, logs, sim.plant, sim.clf, sim.cbfs, plot_config = configuration )
    plotSim.main_ax.set_title("", fontsize=12)
else:
    plotSim = Plot2DSimulation(logs, sim.plant, sim.clf, sim.cbfs, plot_config = configuration)
    plotSim.main_ax.set_title("")

# plotSim.fig.suptitle("Nominal CLF-CBF Controller", fontsize=18)
plotSim.fig.suptitle("Compatible CLF-CBF Controller", fontsize=18)

# Plot 221
time1 = 0.2
configuration["gridspec"] = (1,2,1)
plotSim.configure()
plotSim.main_ax.set_title('Trajectory at t = ' + str(time1) + ' s', fontsize=font_size)
plotSim.plot_frame(time1)

# Subplot 222
time2 = 1.2
configuration["gridspec"] = (1,2,2)
plotSim.configure()
plotSim.main_ax.set_title('Trajectory at t = ' + str(time2) + ' s', fontsize=font_size)
plotSim.plot_frame(time2)

time = logs["time"]

state_x = logs["state"][0]
state_y = logs["state"][1]
all_states = np.hstack([state_x, state_y])

control_x = logs["control"][0]
control_y = logs["control"][1]
all_controls = np.hstack([control_x, control_y])
all_states = np.hstack([state_x, state_y])

max_time = 3.0

# Subplot 323
# ax1.set_aspect('equal', adjustable='box')
# ax1 = plotSim.fig.add_subplot(212)
# ax1.set_title('States', fontsize=font_size)
# ax1.plot(time, state_x, "--", label='$x_1$', linewidth=2, markersize=10)
# ax1.plot(time, state_y, "--", label='$x_2$', linewidth=2, markersize=10)
# ax1.legend(fontsize=14, loc='upper right')
# ax1.set_xlim(0, max_time)
# ax1.set_ylim(np.min(all_states)-1, np.max(all_states)+1)
# ax1.set_xlabel('Time [s]', fontsize=14)
# plt.grid()

# Subplot 324
# ax2.set_aspect('equal', adjustable='box')
# ax2 = plotSim.fig.add_subplot(224)
# ax2.set_title('Control signal', fontsize=font_size)
# ax2.plot(time, control_x, "--", label='$u_1$', linewidth=2, markersize=10, alpha=1.0)
# ax2.plot(time, control_y, "--", label='$u_2$', linewidth=2, markersize=10, alpha=0.6) 
# ax2.legend(fontsize=14, loc='lower right')
# ax2.set_xlim(0, max_time)
# ax2.set_ylim(np.min(all_controls)-1, np.max(all_controls)+1)
# ax2.set_xlabel('Time [s]', fontsize=14)
# plt.grid()

plt.savefig(simulation_file + "_plots.eps", format='eps', transparent=True)
# plt.savefig(simulation_file + ".svg", format="svg",transparent=True)

plt.show()