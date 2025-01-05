import sys, importlib
import matplotlib.pyplot as plt
import numpy as np
from common import optimal_arrangement
from controllers.compatibility import QFunction

sim_config = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples.simulation."+sim_config, package=None)

Qfunctions = [ QFunction(sim.plant, sim.clf, cbf, p=1.0) for cbf in sim.cbfs ]

for k, qfun in enumerate(Qfunctions):

    barrier = qfun.compatibility_barrier()
    lower = qfun.lowerbound_dS_SoS()
    print(f"Compatibility = {barrier}")
    print(f"Lowerbound = {lower}")

''' --------------------------------- Plot ------------------------------------- '''
size = 3.0
nrows, ncols = optimal_arrangement(len(sim.cbfs))

fig = plt.figure(figsize=(1.8*size*ncols, size*nrows), layout='constrained')
fig.suptitle('Q-function')

axes = []
for k, qfun in enumerate(Qfunctions):

    ax = fig.add_subplot(nrows, ncols,1+k)
    axes.append(ax)

    qfun.init_graphics(ax)
    qfun.plot()

plt.show()