import sys, importlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from common import optimal_arrangement

simulation_config = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_config, package=None)

# Simulation loop -------------------------------------------------------------------
num_steps = int(sim.T/sim.sample_time)
time_list = []

''' --------------------------------- Plot ------------------------------------- '''
fig = plt.figure(figsize=(4.0*sim.num_cbfs, 4.0), layout='constrained')
fig.suptitle('Q-functions')

axes = []
nrows, ncols = optimal_arrangement(sim.num_cbfs)
for k, qfun in enumerate(sim.controller.Qfunctions):

    ax = fig.add_subplot(nrows, ncols,1+k)
    axes.append(ax)

    qfun.init_graphics(ax)
    qfun.plot()

plt.show()