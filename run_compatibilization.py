import sys
import importlib

from controllers import CompatibleQP

''' ------------------------- Load simulation example ------------------------------ '''
sim_config = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples.simulation."+sim_config, package=None)

control_opts = {"compatibilization": True, "active": True}

''' ---------------------------- Load controller ---------------------------------- '''
sample_time = 2e-2
controller = CompatibleQP(sim.plant, sim.clf, sim.cbfs, alpha = [1.0, 2.0], beta = 1.0, p = [1.0, 1.0], 
                          dt = sample_time, **control_opts, verbose=True)