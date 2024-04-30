import numpy as np
import scipy as sp
from functions import Kernel

n = 2
initial_state = [0.5, 6.0]

kernel = Kernel(*initial_state, degree=3)
p = kernel.kernel_dim
Amatrices = kernel.get_A_matrices()

A_norms = [ np.linalg.norm(A.T) for A in Amatrices ]
sum_ATA = np.zeros([p,p])
for A in Amatrices:
    sum_ATA += A.T @ A 
# print(A_norms)

MAX_RADIUS = 100

sqrtP = np.random.rand(p,p)
P = sqrtP.T @ sqrtP
Pi_list = [ Ai.T @ P + P @ Ai for Ai in Amatrices ]
eigP = np.linalg.eigvals(P)
diagP = sp.linalg.block_diag(*[ P for _ in range(n) ])

SOSConvex = np.block([ [ Ai.T @ P @ Aj + Ai.T @ Aj.T @ P for Aj in Amatrices ] for Ai in Amatrices ])    
eigSOSConvex = np.linalg.eigvals(SOSConvex)

print(f"P spectra = {eigP}")
# print(f"SOS Convex spectra = {eigSOSConvex}")

N = 10000
# x = np.zeros(n)
AmmA_list = []
AmmA_norms = []
AmmA = 100*np.ones([p*n, p*n])
print(f"||AmmA|| = {np.linalg.norm(AmmA)}")
for k in range(N):

    r = MAX_RADIUS*np.random.rand()
    theta = np.pi*np.random.rand()

    x = np.array([ r*np.cos(theta), r*np.sin(theta) ])
    m = kernel.function(x)   
    AmmA = (1/(m.T @ m))*np.block([ [ Ai @ np.outer(m,m) @ Aj.T for Aj in Amatrices ] for Ai in Amatrices ])

    print(f"||AmmA|| = {np.linalg.norm(AmmA)}")

    # if np.all(np.linalg.eigvals(AmmA - newAmmA) > 0.0): AmmA = newAmmA



    # if not is_min_psd:
    #     print("Hypothesis is wrong")
    #     break