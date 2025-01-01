import sys
import importlib

from controllers.compatibility import QFunction

simulation = 'LTI1_multiple'
if len(sys.argv) > 1:
    simulation = sys.argv[1]

file_path = "examples.simulation." + simulation
sim = importlib.import_module(file_path, package=None)

qfun = QFunction(sim.plant, sim.clf, sim.cbf1, p=1.0)
qfun._init_compatible_opt()

qfun.solve_compatibility_opt()