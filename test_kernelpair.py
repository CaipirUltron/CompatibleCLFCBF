import sys
import importlib

import numpy as np
import contourpy as ctp
import matplotlib.pyplot as plt

from functions import KernelPair

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Invariant set plot for Kernel-based CLF-CBFs")
ax.set_aspect('equal', adjustable='box')

limits = sim.plot_config["limits"]

clf_cbf_pair = KernelPair(sim.clf, sim.cbf, sim.plant, params={"slack_gain": sim.p, "clf_gain": sim.alpha})
clf_cbf_pair.plot_invariant(ax=ax, limits=sim.plot_config["limits"], spacing=0.2, extended=False)

plt.show()