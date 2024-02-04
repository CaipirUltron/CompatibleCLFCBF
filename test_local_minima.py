import sys
import importlib

import numpy as np
import matplotlib.pyplot as plt

from controllers.equilibrium_algorithms import compute_equilibria, plot_invariant, closest_to_image

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

P, Q = sim.clf.P, sim.cbf.Q
A_list = sim.kernel.get_A_matrices()
n = len(A_list)

print(f"Eigenvalues of P = {np.linalg.eigvals(P)}")
print(f"Eigenvalues of Q = {np.linalg.eigvals(Q)}")

P_list = [ A.T @ P + P @ A for A in A_list ]
Q_list = [ A.T @ Q + Q @ A for A in A_list ]

for k in range(n):
    Pi = P_list[k]
    Qi = Q_list[k]
    print(f"Eigenvalues of P{k+1} = {np.linalg.eigvals(Pi)}")
    print(f"Eigenvalues of Q{k+1} = {np.linalg.eigvals(Qi)}")

    # A = A_list[k]
    # print(f"Commutator of A.T P = \n{(A.T @ P - P @ A)}")
    # print(f"Commutator of A.T Q = \n{(A.T @ Q - Q @ A)}")