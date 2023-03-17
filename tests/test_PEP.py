'''
Tests the solution of the parameter eigenvalue problem.
'''

import numpy as np
from controllers.compatibility import solve_PEP

n = 2

A = np.random.rand(n,n)
A = A.T @ A

B = np.random.rand(n,n)
B = B.T @ B

C = np.random.rand(n,n)
C = C.T @ C

const = 1
mu1, mu2, Z = solve_PEP( A, B, C, constant = const, mu2 = np.random.rand(), max_iter = 10000 )

F = np.zeros(n)
error_pencil = np.ones([n,n])
error_mu2 = np.zeros(n)
error_boundary = np.zeros(n)
for k in range(n):

    L = (mu1[k] * A - mu2[k] * B + C)

    error_pencil[:,k] = L @ Z[:,k]
    error_mu2[k] = mu2[k] - 0.5 * const * Z[:,k] @ B @ Z[:,k]
    error_boundary[k] = Z[:,k] @ A @ Z[:,k] - 1


print("Error pencil = " + str(error_pencil))
print("Error mu2 = " + str(error_mu2))
print("Error boundary = " + str(error_boundary))