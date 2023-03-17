'''
Checks if the pencil eigenvalues are being computed correctly.
'''

import numpy as np
from controllers.compatibility import LinearMatrixPencil

n = 6

A = np.random.rand(n,n)
A = A.T @ A

B = np.random.rand(n,n)
B = B.T @ B

pencil = LinearMatrixPencil( A, B )
eigs = pencil.eigenvalues
Z = pencil.eigenvectors

error = np.random.rand(n,n)
for k in range(len(eigs)):
    error[:,k] = pencil.value( eigs[k] ) @ Z[:,k]

print(error)