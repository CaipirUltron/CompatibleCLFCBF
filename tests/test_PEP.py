import numpy as np
from controllers.compatibility import solve_PEP, compute_mu

n = 2

A = np.random.rand(n,n)
A = A.T @ A

B = np.random.rand(n,n)
B = B.T @ B

C = np.random.rand(n,n)
C = C.T @ C

mu2 = np.random.rand(n)

const = 1
mu1, mu2, Z = solve_PEP( A, B, C, const, mu2 )

error_mu1 = np.zeros(n)
error_mu2 = np.zeros(n)
error_Z = np.ones([n,n])
for k in range(n):

    L = (mu1[k] * A + mu2[k] * B + C)

    eigs, Q = np.linalg.eig(L)
    print("Eigenvalues of L = " + str(eigs))
    for k in range(len(eigs)):
        if np.abs(eigs[k]) < 0.00001:
            error_Z[:,k] = Z[:,k] - Q[:,k]
            break

    error_mu2[k] = compute_mu( B, const, Z[:,k] ) - mu2[k]

print("Error Z = " + str(error_Z))
print("Error mu2 = " + str(error_mu2))