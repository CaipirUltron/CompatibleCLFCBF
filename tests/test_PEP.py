'''
Tests the computation of generalized eigenvalues with Affine Matrix Pencils
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from controllers.compatibility import solve_PEP1, solve_PEP2
from examples.integrator_nominalQP import clf_params, cbf_params1

Hv = clf_params["Hv"]
x0 = clf_params["x0"]

Hh = cbf_params1["Hh"]
p0 = cbf_params1["p0"]

n = len(Hv)

temp_P = -(Hv @ x0).reshape(n,1)
P = np.block([ [ Hv       , temp_P       ], 
               [ temp_P.T , x0 @ Hv @ x0 ] ])

temp_Q = -(Hh @ p0).reshape(n,1)
Q = np.block([ [ Hh       , temp_Q       ], 
               [ temp_Q.T , p0 @ Hh @ p0 ] ])

C = sp.linalg.block_diag(np.zeros([n,n]), 1)

dim = 3

# P = np.random.rand(dim,dim)
# P = P.T @ P
# print("P = " + str(P))

# Q = np.random.rand(dim,dim)
# Q = Q.T @ Q
# print("Q = " + str(Q))

selection = np.zeros(dim)
selection[-1] = 1

lambda2_0 = -10*np.random.rand()
lambda1, lambda2, Z, lambda1_list, lambda2_list = solve_PEP1( Q, P, init_lambda = lambda2_0, max_iter = 1000 )

fig = plt.figure()
for k in range(len(lambda1_list)):

    L = (lambda1[k] * Q - lambda2[k] * C - P)
    z = Z[:,k]
    
    error_pencil = np.linalg.norm(L @ z)
    error_lambda = lambda1[k] - lambda2[k] - z @ P @ z
    last_kernel_elem = z @ C @ z
    error_boundaries = z @ Q @ z - 1

    phrase1 = str(k+1) + "-th solution: \n"
    phrase2 = "Lambda = " + str(lambda1[k]) + "\n"
    phrase3 = "Kappa = " + str(lambda2[k]) + "\n"
    phrase4 = "with eigenvector = " + str(Z[:,k].T)
    print(phrase1+phrase2+phrase3+phrase4)

    print("Error pencil = " + str(error_pencil))
    print("Error lambdas = " + str(error_lambda))
    print("Last kernel element = " + str(last_kernel_elem))
    print("Error boundary = " + str(error_boundaries))

    plt.scatter( lambda2_list[k], lambda1_list[k], linewidth=0.2, marker='o' )

plt.show()