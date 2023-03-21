'''
Tests the solution of the parameter eigenvalue problem for realistic problems.
'''
import numpy as np
import scipy as sp
from controllers.compatibility import solve_PEP
from examples.integrator_nominalQP import clf_params, cbf_params1

# Hv = clf_params["Hv"]
# x0 = clf_params["x0"]

# Hh = cbf_params1["Hh"]
# p0 = cbf_params1["p0"]

# n = len(Hv)

# temp_P = -(Hv @ x0).reshape(n,1)
# P = np.block([ [ Hv       , temp_P       ], 
#                [ temp_P.T , x0 @ Hv @ x0 ] ])

# temp_Q = -(Hh @ p0).reshape(n,1)
# Q = np.block([ [ Hh       , temp_Q       ], 
#                [ temp_Q.T , p0 @ Hh @ p0 ] ])

# C = sp.linalg.block_diag(np.zeros([n,n]), 1)

n = 3

P = np.random.rand(n,n)
P = P.T @ P

Q = np.random.rand(n,n)
Q = Q.T @ Q

C = np.random.rand(n,n)
C = C.T @ C

selection = np.zeros(n)
selection[-1] = 1

const = 1
mu1, mu2, Z = solve_PEP( Q, P, C, constant = const, mu2 = np.random.rand(), max_iter = 10000, step = 1.0 )

F = np.zeros(n)
error_pencil = np.ones([n,n])
error_mu2 = np.zeros(n)
error_kernel_image = np.zeros(n)
error_boundaries = np.zeros(n)
for k in range(n):

    L = (mu1[k] * Q - mu2[k] * P + C)

    error_pencil[:,k] = L @ Z[:,k]
    error_mu2[k] = mu2[k] - 0.5 * const * Z[:,k] @ P @ Z[:,k]
    error_kernel_image[k] = selection @ Z[:,k] - 1
    error_boundaries[k] = Z[:,k] @ Q @ Z[:,k] - 1

print("Error pencil = " + str(error_pencil))
print("Error mu2 = " + str(error_mu2))
print("Error kernel image = " + str(error_kernel_image))
print("Error boundary = " + str(error_boundaries))