import json, importlib
import matplotlib.pyplot as plt

from graphics import PlotQuadraticSim

simulation = "LTI2_multiple"
file_names = [simulation+"_nominal", simulation+"_compatible"]

nom_times = [ 0.0, 5, 9.9 ]
# com_times = [ 0.2, 0.4, 1.0 ]
com_times = [ 0.2, 1.0, 3.0 ]

num_files = len(file_names)
num_times = len(nom_times)

hor_size, ver_size = 7.0, 4.0
fig, ax = plt.subplots(nrows=num_files, ncols=num_times, figsize=(hor_size, ver_size), 
                       constrained_layout=False,
                       sharex=True,
                       sharey=True)

fig.suptitle('Nominal vs Compatible CLF-CBF QP-controller')
fig.tight_layout(pad=0.1)

for f_index, file_name in enumerate(file_names):
    if "nominal" in file_name:
        times = nom_times
    elif "compatible" in file_name:
        times = com_times
    for t_index, time in enumerate(times):

        file_path = "examples.simulation." + simulation
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