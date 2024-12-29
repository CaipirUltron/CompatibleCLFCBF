import sys, importlib
import matplotlib.pyplot as plt

from common import optimal_arrangement
from controllers import CompatibleQP
from controllers.compatibility import QFunction

sim_config = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+sim_config, package=None)


Qfunctions = [ QFunction(sim.plant, sim.clf, cbf, p=1.0) for cbf in sim.cbfs ]

''' --------------------------------- Plot ------------------------------------- '''
size = 4.0
fig = plt.figure(figsize=(1.5*size*len(sim.cbfs), size), layout='constrained')
fig.suptitle('Q-function')

axes = []
nrows, ncols = optimal_arrangement(len(sim.cbfs))
for k, qfun in enumerate(Qfunctions):

    ax = fig.add_subplot(nrows, ncols,1+k)
    axes.append(ax)

    qfun.init_graphics(ax)
    qfun.plot()

plt.show()