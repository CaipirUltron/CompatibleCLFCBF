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
from scipy.linalg import null_space
from controllers.compatibility import solve_PEP, PolynomialCLFCBFPair
from examples.integrator_nominalQP import clf_params, cbf_params1, cbf_params2, cbf_params3

is_from_simulation = True
cbf_params = cbf_params2

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
    loaded = 1
    Hv, Hh = clf_params["Hv"], cbf_params["Hh"]
    x0, p0 = clf_params["x0"], cbf_params["p0"]

    n = len(Hv)
    dim = n + 1

    temp_P = -(Hv @ x0).reshape(n,1)
    P = np.block([[ Hv       , temp_P       ], 
                  [ temp_P.T , x0 @ Hv @ x0 ]])
    null_space_P = np.hstack([ x0, 1.0 ])

    temp_Q = -(Hh @ p0).reshape(n,1)
    Q = np.block([[ Hh       , temp_Q       ], 
                  [ temp_Q.T , p0 @ Hh @ p0 ]])
    null_space_Q = np.hstack([ p0, 1.0 ])

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

# Create polynomial CLF-CBF pair -----------------------------------------------------------------
pair = PolynomialCLFCBFPair(P, Q)

# Initialize graph -------------------------------------------------------------------------------
kappa_list, lambda_list = np.linspace(-200, 200, 500), np.linspace(-20, 20, 500)
K, L = np.meshgrid( kappa_list, lambda_list )

def compute_det(lambda_p, kappa_p):
    L = lambda_p * pair.Q - kappa_p * pair.C - pair.P
    return np.linalg.det(L)
det = np.frompyfunc(compute_det, 2, 1)

fig, ax = plt.subplots(tight_layout=False)

# Plot level sets -------------------------------------------------------------------------------
cp = ax.contour(K, L, det( L, K ), 0, colors='black')

# Plot asymptotes -------------------------------------------------------------------------------
for m in pair.asymptotes.keys():
    print(str(m) + ": " + str(pair.asymptotes[m]) + "\n")
    for p in pair.asymptotes[m]:
        if np.abs(m) == np.inf:
            ax.plot( [p, p], [ lambda_list[0], lambda_list[-1]], '--', color='green' )
            continue
        ax.plot( [kappa_list[0], kappa_list[-1]], [ m*kappa_list[0]+p, m*kappa_list[-1]+p], '--', color='green' )

# Plot level sets and initial line --------------------------------------------------------------
init_line = { "angular_coef":  -0.05, "linear_coef" : 4.0 }
m, p = init_line["angular_coef"], init_line["linear_coef"]
ax.plot( [kappa_list[0], kappa_list[-1]], [ m*kappa_list[0]+p, m*kappa_list[-1]+p], '--', color='red' )

# Solve PEP and test results --------------------------------------------------------------------
t = time.time()
lambda_p, kappa_p, Z = solve_PEP( pair.Q, pair.P, initial_line = init_line, max_iter = 1000 )
print( "Elapsed " + str(time.time() - t) + " seconds." )

for k in range(len(lambda_p)):

    L = ( lambda_p[k] * pair.Q - kappa_p[k] * pair.C - pair.P )
    z = Z[:,k]

    error_pencil = np.linalg.norm(L @ z)
    last_kernel_elem = z @ pair.C @ z
    error_boundaries = z @ pair.Q @ z - 1

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

test_vars = {"P": pair.P.tolist(), "Q": pair.Q.tolist()}
if (~is_from_simulation) and (loaded == 0):
    print("Save file? Y/N")
    if str(input()).lower() == "y":
        print("File name: ")
        file_name = str(input())
        with open(file_name+".json", "w") as file:
            print("Saving test data...")
            json.dump(test_vars, file, indent=4)