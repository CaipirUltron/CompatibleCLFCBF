import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

from common import rgb
from controllers.equilibrium_algorithms import check_equilibrium, compute_equilibria, closest_compatible, angle2boundary

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

Q = sim.cbf.Q
rankQ = np.linalg.matrix_rank(Q)
dim_elliptical_manifold = rankQ - 1

limits = [ [-10, 10], [5, 10] ]
init_theta = np.random.uniform(0, 2*np.pi, dim_elliptical_manifold)
x = angle2boundary(sim.cbf, init_theta, limits = limits)

if x != None:
    m = sim.kernel.function(x)
    print("mQm = " + str(m.T @ Q @ m))
print("Boundary point = " + str(x))
