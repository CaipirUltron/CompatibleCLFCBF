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
    with open(simulation_file + ".json") as file:
        print("Loading graphical simulation with " + simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate "+simulation_file + ".json")

font_size = 14
configuration = {
    "figsize": (8,8),
    "gridspec": (2,2,1),
    "widthratios": [1, 1],
    "heightratios": [1.0, 1.0],
    "axeslim": (-9,9,-9,9),
    "path_length": 10,
    "numpoints": 1000,
    "drawlevel": True,
    "resolution": 50,
    "fps":50,
    "pad":2.0,
    "equilibria": False
}

if hasattr(sim, 'path'):
    plotSim = PlotPFSimulation( sim.path, logs, sim.plant, sim.clf, sim.cbfs, plot_config = configuration )
    plotSim.main_ax.set_title("", fontsize=12)
else:
    plotSim = Plot2DSimulation(logs, sim.plant, sim.clf, sim.cbfs, plot_config = configuration)
    plotSim.main_ax.set_title("")

plotSim.fig.suptitle("Path Following with Obstacle Avoidance using Compatible CLF-CBFs", fontsize=18)

# Plot 221
configuration["gridspec"] = (2,2,1)
plotSim.configure()
plotSim.main_ax.set_title("")
plotSim.plot_frame(1.92)

# Subplot 222
configuration["gridspec"] = (2,2,2)
plotSim.configure()
plotSim.main_ax.set_title("")
plotSim.plot_frame(3.1)

# Subplot 223
configuration["gridspec"] = (2,2,3)
plotSim.configure()
plotSim.main_ax.set_title("")
plotSim.plot_frame(4.1)

# Subplot 224
configuration["gridspec"] = (2,2,4)
plotSim.configure()
plotSim.main_ax.set_title("")
plotSim.plot_frame(6.0)

time = logs["time"]

state_x = logs["state"][0]
state_y = logs["state"][1]
all_states = np.hstack([state_x, state_y])

control_x = logs["control"][0]
control_y = logs["control"][1]
all_controls = np.hstack([control_x, control_y])

gamma = logs["gamma_log"]
error_x, error_y = [], []
for i in range(len(gamma)):
    pt = sim.path.get_path_point( gamma[i] )
    error_x.append( state_x[i] - pt[0] )
    error_y.append( state_y[i] - pt[1] )
all_errors = np.hstack([error_x, error_y])

plt.savefig(simulation_file + "_traj.eps", format='eps', transparent=True)

max_time = 7.0
fig = plt.figure(figsize = (6,6), constrained_layout=True)

# Subplot 325
ax1 = fig.add_subplot(211)
# ax1.set_aspect('equal', adjustable='box')
ax1.set_title('Path following error', fontsize=font_size)
ax1.plot(time, error_x, "--", label='$e_1$', linewidth=2, markersize=10)
ax1.plot(time, error_y, "--", label='$e_2$', linewidth=2, markersize=10)
ax1.legend(fontsize=14, loc='upper right')
ax1.set_xlim(0, max_time)
ax1.set_ylim(np.min(all_errors)-1, np.max(all_errors)+1)
# ax1.set_xlabel('Time [s]', fontsize=14)
plt.grid()

# Subplot 326
ax2 = fig.add_subplot(212)
# ax2.set_aspect('equal', adjustable='box')
ax2.set_title('Control signal', fontsize=font_size)
ax2.plot(time, control_x, "--", label='$u_1$', linewidth=2, markersize=10, alpha=1.0)
ax2.plot(time, control_y, "--", label='$u_2$', linewidth=2, markersize=10, alpha=0.6) 
ax2.legend(fontsize=14, loc='upper right')
ax2.set_xlim(0, max_time)
ax2.set_ylim(np.min(all_controls)-1, np.max(all_controls)+1)
ax2.set_xlabel('Time [s]', fontsize=14)
plt.grid()

plt.savefig(simulation_file + "_plots.eps", format='eps', transparent=True)
# plt.savefig(simulation_file + ".svg", format="svg",transparent=True)

plt.show()