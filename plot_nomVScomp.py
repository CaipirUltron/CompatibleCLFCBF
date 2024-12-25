import sys, json, importlib
import matplotlib.pyplot as plt

from itertools import product
from graphics import PlotQuadraticSim

file_names = ["LTI_multiple_nominal", "LTI_multiple_compatible"]
times = [ 0.0, 6.8, 7.5 ]

num_files = len(file_names)
num_times = len(times)

hor_size, ver_size = 7.0, 4.0
fig, ax = plt.subplots(nrows=num_files, ncols=num_times, 
                       figsize=(hor_size, ver_size), 
                       constrained_layout=False,
                       sharex=True,
                       sharey=True)

fig.suptitle('Nominal vs Compatible CLF-CBF QP-controller')
fig.tight_layout(pad=0.1)

for f_index, file_name in enumerate(file_names):
    for t_index, time in enumerate(times):

        file_path = "examples." + file_name
        sim = importlib.import_module(file_path, package=None)

        try:
            with open("logs/"+file_name+".json") as file:
                print("Loading graphical simulation with "+file_name+".json")
                logs = json.load(file)
        except IOError:
            print("Couldn't locate " + file_name + ".json")

        curr_ax = ax[f_index, t_index]

        if f_index == 0 and t_index == 0:
            curr_ax.set_ylabel("nominal QP")
        if f_index == 1 and t_index == 0:
            curr_ax.set_ylabel("compatible QP")


        plotSim = PlotQuadraticSim( logs, sim.plant, sim.clf, sim.cbfs, plot_config=sim.plot_config )
        plotSim.init_graphics(curr_ax)
        plotSim.plot_frame(time)

plt.show()