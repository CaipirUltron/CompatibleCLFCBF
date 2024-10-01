import numpy as np
import sys, importlib
import matplotlib.pyplot as plt

from functions import Kernel

simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

''' Test invariant pencil '''

N = 1000
total_errors = [ 0.0 for _ in range(len(sim.cbfs)) ]


for cbf_index in range(len(sim.cbfs)):

    A, B, kernel_powers = sim.kerneltriplet.invariant_pencil( cbf_index )
    kernel = Kernel(dim=sim.n, monomials=kernel_powers)
    
    for k in range(N):

        x = np.random.randn(sim.n)
        l = np.random.rand()

        vQ = sim.kerneltriplet.vecQ_fun(x, cbf_index)
        vP = sim.kerneltriplet.vecP_fun(x)
        val1 = l * vQ - vP
        val2 = ( l * A - B ) @ kernel.function(x)

        total_errors[cbf_index] += np.linalg.norm( val1 - val2 )

for k, err in enumerate(total_errors):
    print(f"Total error for CBF{k+1} = {err}")