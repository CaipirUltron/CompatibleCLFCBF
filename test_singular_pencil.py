import importlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import platform, os, sys, json, time
import control

# simulation_config = sys.argv[1].replace(".json","")
# sim = importlib.import_module("examples."+simulation_config, package=None)

# ------------------------------------ Define kernel -----------------------------------
# A, B, kernel = sim.kerneltriplet.invariant_pencil()
# print(f"kernel = {kernel}")

n, p = (2, 2)
A = np.random.randn(n,n)
B = np.random.randn(n,n)

print(f"A = {A}")
print(f"B = {B}")

Controllability = control.ctrb(A, B)
rank = np.linalg.matrix_rank( Controllability )
if rank < A.shape[0]:
    print(f"(A, B) pair is not controllable.")
else:
    print(f"(A, B) pair is controllable.")
    K = control.acker(A, B, poles=[ -2, -2 ])
    Al = A - B @ K

    eigsAl = np.linalg.eigvals(Al)
    print(f"eig(Al) = {eigsAl}")

n,p = A.shape
Q = np.random.randn(p,p)
Q = Q.T @ Q
c = np.random.randn(n)

''' ---------------------- Find general solutions of ( l A - B ) z = c ---------------- '''

max_order = 100
for deg in range(0, max_order):

    N = np.zeros([ (deg+2)*n, (deg+1)*p ])
    for i in range(deg+1):
        N[ i*n:(i+1)*n , i*p:(i+1)*p ] = - B
        N[ (i+1)*n:(i+2)*n , i*p:(i+1)*p ] = + A

    Null = sp.linalg.null_space(N)

    if Null.size != 0:
        break

# print(f"nullspace polynomial degree = {deg}")
# print(f"N shape = {N.shape}")
# print(f"null space shape= {Null.shape}")

# lambda_min = -1000
# lambda_max = 1000
# lambda_res = 0.01
# lambdas = np.arange(lambda_min, lambda_max, lambda_res)
# num_pts = len(lambdas)

def compute_terms(beta: np.ndarray):
    ''' zQz as a function of the beta parameters '''

    zQz = np.zeros(num_pts)
    zH_Q_zH = np.zeros(num_pts)
    zQz_mix = np.zeros(num_pts)
    zP_Q_zP = np.zeros(num_pts)

    if len(beta) != Null.shape[1]:
        raise TypeError("Beta size is incorrect")

    for k, l in enumerate(lambdas):

        Lambda = np.hstack([ (l**power) * np.eye(p) for power in range(deg+1) ])

        zH = np.zeros(p)
        for i in range(len(beta)):
            zH += beta[i] * Lambda @ Null[:,i]

        pinvAB = ( l * A - B ).T @ np.linalg.inv ( ( l * A - B ) @ ( l * A - B ).T )
        zP = pinvAB @ c
        z = zH + zP

        zQz[k] = z.T @ Q @ z
        zH_Q_zH[k] = zH.T @ Q @ zH
        zQz_mix[k] = zH.T @ Q @ zP + zP.T @ Q @ zH
        zP_Q_zP[k] = zP.T @ Q @ zP

    return zQz, zH_Q_zH, zQz_mix, zP_Q_zP

# beta = np.random.randn(Null.shape[1])
# zQz, zH_Q_zH, zQz_mix, zP_Q_zP = compute_terms(beta)

# # ------------------------------------ Plot -----------------------------------

# fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10.0, 10.0), layout="constrained")
# fig.suptitle('Test Singular Pencil')

# ax[0,0].plot( lambdas, zQz, label='zQz' )
# ax[0,1].plot( lambdas, zH_Q_zH, label='zH_Q_zH' )
# ax[1,0].plot( lambdas, zQz_mix, label='zQz_mix' )
# ax[1,1].plot( lambdas, zP_Q_zP, label='zP_Q_zP' )

# for i in range(2): 
#     for j in range(2):
#         ax[i,j].legend()
#         ax[i,j].set_xlim(lambda_min, lambda_max) 

plt.show()