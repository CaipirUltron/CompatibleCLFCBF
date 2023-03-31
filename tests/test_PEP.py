'''
Tests the computation of generalized eigenvalues with two-parameter Affine Matrix Pencils
'''
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import numpy as np
import scipy as sp
from controllers.compatibility import solve_PEP
from examples.integrator_nominalQP import clf_params, cbf_params1, cbf_params2, cbf_params3

cbf_params = cbf_params2

Hv, Hh = clf_params["Hv"], cbf_params["Hh"]
x0, p0 = clf_params["x0"], cbf_params["p0"]
n = len(Hv)

temp_P = -(Hv @ x0).reshape(n,1)
P = np.block([ [ Hv       , temp_P       ], 
               [ temp_P.T , x0 @ Hv @ x0 ] ])
temp_Q = -(Hh @ p0).reshape(n,1)
Q = np.block([ [ Hh       , temp_Q       ], 
               [ temp_Q.T , p0 @ Hh @ p0 ] ])
C = sp.linalg.block_diag(np.zeros([n,n]), 1)

dim = n+1

selection = np.zeros(dim)
selection[-1] = 1

# Plot level sets and initial line ---------------------------------------------------------------
init_line = {"angular_coef": -0.2,
             "linear_coef" :  3.0 }

m = init_line["angular_coef"]
p = init_line["linear_coef"]

def compute_det(lambda_p, kappa_p):
    L = lambda_p * Q - kappa_p * C - P
    return np.linalg.det(L)

det = np.frompyfunc(compute_det, 2, 1)

kappa_list = np.linspace(-100, 100, 200)
lambda_list = np.linspace(-20, 20, 200)
K, L = np.meshgrid( kappa_list, lambda_list )

func = det( L, K )

fig, ax = plt.subplots(tight_layout=True)
cp = ax.contour(K, L, func, 0, colors='black')
# plt.clabel(cp, inline=True, fontsize=8)

ax.plot( [kappa_list[0], kappa_list[-1]], [ m*kappa_list[0]+p, m*kappa_list[-1]+p], '--', color='green' )

ax.set_xlim(kappa_list[0], kappa_list[-1])
ax.set_ylim(lambda_list[0], lambda_list[-1])
ax.set_xlabel("$\kappa$")
ax.set_ylabel("$\lambda$")
ax.set_title("$\det( \lambda Q - \kappa C - P )$")

# Solve PEP and test results --------------------------------------------------------------------
lambda_p, kappa_p, Z = solve_PEP( Q, P, initial_line = init_line, max_iter = 1000 )

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

plt.show()