import numpy as np
import sys, importlib
import matplotlib.pyplot as plt

from functions import Kernel

simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

N = 1000
total_errors = [ 0.0 for _ in range(len(sim.cbfs)) ]
segment_errors = [ [] for _ in range(len(sim.cbfs)) ]
for cbf_index in range(len(sim.cbfs)):

    A, B, kernel_powers = sim.kerneltriplet.invariant_pencil( cbf_index )
    kernel = Kernel(dim=sim.n, monomials=kernel_powers)
    print(kernel)

    ''' ---------------- Test invariant pencil at random points ------------------- '''

    for k in range(N):

        print(f"Testing with {k+1} sample...")

        x = np.random.randn(sim.n)
        l = np.random.rand()

        vQ = sim.kerneltriplet.vecQ_fun(x, cbf_index)
        vP = sim.kerneltriplet.vecP_fun(x)
        val1 = l * vQ - vP
        val2 = ( l * A - B ) @ kernel.function(x)

        total_errors[cbf_index] += np.linalg.norm( val1 - val2 )

    ''' -------------- Test invariant pencil at segment points -------------------- '''

    cbf_segments = sim.kerneltriplet.invariant_segs[cbf_index]
    segment_errors[cbf_index] = [ 0.0 for _ in range(len(cbf_segments)) ]
    for seg_index, seg in enumerate(cbf_segments):
        for l, pt in zip(seg["lambdas"], seg["points"]):
            invariant_vec = ( l * A - B ) @ kernel.function(pt)
            segment_errors[cbf_index][seg_index] += np.linalg.norm( invariant_vec )

''' -------------------- Print results ------------------------- '''

for cbf_index, err in enumerate(total_errors):
    print(f"Total error for CBF{cbf_index+1} = {err}")

for cbf_index, seg_errors in enumerate(segment_errors):
    print(f"Segment errors for CBF{cbf_index+1}: \n")
    for k, seg_error in enumerate(seg_errors):
        print(f"segment {k+1} error = {seg_error}")