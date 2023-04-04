'''
Tests the computation of generalized eigenvalues with two-parameter Affine Matrix Pencils.
Can test with known matrices from simulations or random matrices (generated at runtime). 
Can also save test matrices for latter testing. 
'''
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import sys
import json
import time
import numpy as np
import scipy as sp
from controllers.compatibility import solve_PEP, LinearMatrixPencil2
from examples.integrator_nominalQP import clf_params, cbf_params1, cbf_params2, cbf_params3

is_from_simulation = False

# Generate random matrices ------------------------------------------------------------------------
n = 2
dim = n + 1

P = np.random.rand(dim,dim)
P = P.T @ P
eigP , _= np.linalg.eig(P)
P = P/np.max(eigP)
Q = np.random.rand(dim,dim)
Q = Q.T @ Q
eigQ , _= np.linalg.eig(Q)
Q = Q/np.max(eigQ)

# Use simulation matrices or previously saved ones ------------------------------------------------
loaded = 0
if is_from_simulation:
    cbf_params = cbf_params1
    Hv, Hh = clf_params["Hv"], cbf_params["Hh"]
    x0, p0 = clf_params["x0"], cbf_params["p0"]

    n = len(Hv)
    dim = n + 1

    temp_P = -(Hv @ x0).reshape(n,1)
    P = np.block([[ Hv       , temp_P       ], 
                  [ temp_P.T , x0 @ Hv @ x0 ]])
    temp_Q = -(Hh @ p0).reshape(n,1)
    Q = np.block([[ Hh       , temp_Q       ], 
                  [ temp_Q.T , p0 @ Hh @ p0 ]])
elif len(sys.argv) > 1:
    loaded = 1
    test_config = sys.argv[1].replace(".json","")
    with open(test_config + ".json") as file:
        print("Loading test: " + test_config + ".json")
        test_vars = json.load(file)
    P = np.array(test_vars["P"])
    Q = np.array(test_vars["Q"])
    dim = np.shape(P)[0]
    n = dim - 1

C = sp.linalg.block_diag(np.zeros([n,n]), 1)

# Initialize graph -------------------------------------------------------------------------------
kappa_list, lambda_list = np.linspace(-20, 20, 500), np.linspace(-20, 20, 500)
K, L = np.meshgrid( kappa_list, lambda_list )

def compute_det(lambda_p, kappa_p):
    L = lambda_p * Q - kappa_p * C - P
    return np.linalg.det(L)
det = np.frompyfunc(compute_det, 2, 1)

fig, ax = plt.subplots(tight_layout=False)

# Plot level sets -------------------------------------------------------------------------------
cp = ax.contour(K, L, det( L, K ), 0, colors='black')

# Plot asymptotes -------------------------------------------------------------------------------
eigvalsQ_red, _ = np.linalg.eig(Q[0:-1,0:-1])
eigvalsP_red, _ = np.linalg.eig(P[0:-1,0:-1])

max_eigvalsQ_red, min_eigvalsQ_red = np.max(eigvalsQ_red), np.min(eigvalsQ_red)
max_eigvalsP_red, min_eigvalsP_red = np.max(eigvalsP_red), np.min(eigvalsP_red)

# Horizontal asymptotes:
lambda_inf_min = min_eigvalsP_red/max_eigvalsQ_red
lambda_inf_max = max_eigvalsP_red/min_eigvalsQ_red

# ax.plot( [kappa_list[0], kappa_list[-1]], [ lambda_inf_min, lambda_inf_min ], '--', color='red' )
# ax.plot( [kappa_list[0], kappa_list[-1]], [ lambda_inf_max, lambda_inf_max ], '--', color='red' )

# General asymptotes
pencil = LinearMatrixPencil2(Q, C)
asymptote_angular_coefs = pencil.eigenvalues
m, p = np.max(asymptote_angular_coefs), 0.0
print(asymptote_angular_coefs)
ax.plot( [ kappa_list[0], kappa_list[-1] ], [ m*kappa_list[0]+p, m*kappa_list[-1]+p ], '--', color='red' )

# Plot level sets and initial line -----------------------------------------------------------
init_line = { "angular_coef":  m, "linear_coef" : -p }
m, p = init_line["angular_coef"], init_line["linear_coef"]
# ax.plot( [kappa_list[0], kappa_list[-1]], [ m*kappa_list[0]+p, m*kappa_list[-1]+p], '--', color='green' )

# Solve PEP and test results --------------------------------------------------------------------
t = time.time()
lambda_p, kappa_p, Z = solve_PEP( Q, P, initial_line = init_line, max_iter = 1000 )
print( "Elapsed " + str(time.time() - t) + " seconds." )

for k in range(len(lambda_p)):

    L = (lambda_p[k] * Q - kappa_p[k] * C - P)
    z = Z[:,k]

    error_pencil = np.linalg.norm(L @ z)
    last_kernel_elem = z @ C @ z
    error_boundaries = z @ Q @ z - 1

    phrase1 = str(k+1) + "-th solution with lambda = " + str(lambda_p[k]) + "\n"
    phrase2 = "Kappa = " + str(kappa_p[k]) + "\n"
    phrase3 = "with eigenvector = " + str(z)
    print(phrase1+phrase2+phrase3)

    print("Error pencil = " + str(error_pencil))
    print("Last kernel element = " + str(last_kernel_elem))
    print("Error boundary = " + str(error_boundaries))

    ax.plot( kappa_p[k], lambda_p[k], marker='o' )
    ax.text( kappa_p[k], lambda_p[k], str(k+1))

ax.set_xlim(kappa_list[0], kappa_list[-1])
ax.set_ylim(lambda_list[0], lambda_list[-1])
ax.set_xlabel("$\kappa$")
ax.set_ylabel("$\lambda$")
ax.set_title("$\det( \lambda Q - \kappa C - P ) = 0$")

plt.show()

test_vars = {"P": P.tolist(), "Q": Q.tolist()}
if (~is_from_simulation) and (loaded == 0):
    print("Save file? Y/N")
    if str(input()).lower() == "y":
        print("File name: ")
        file_name = str(input())
        with open(file_name+".json", "w") as file:
            print("Saving test data...")
            json.dump(test_vars, file, indent=4)