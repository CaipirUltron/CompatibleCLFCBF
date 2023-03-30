'''
Tests the computation of generalized eigenvalues with Affine Matrix Pencils
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from controllers.compatibility import LinearMatrixPencil2
from examples.integrator_nominalQP import clf_params, cbf_params1, cbf_params2, cbf_params3

cbf_params = cbf_params2

Hv = clf_params["Hv"]
x0 = clf_params["x0"]
print("CLF minimum = " + str(x0))

Hh = cbf_params["Hh"]
p0 = cbf_params["p0"]
print("CBF minimum = " + str(p0))

n = len(Hv)

temp_P = -(Hv @ x0).reshape(n,1)

P = np.block([ [ Hv       , temp_P       ], 
               [ temp_P.T , x0 @ Hv @ x0 ] ])

temp_Q = -(Hh @ p0).reshape(n,1)
Q = np.block([ [ Hh       , temp_Q       ], 
               [ temp_Q.T , p0 @ Hh @ p0 ] ])

n += 1

#########################################################################

# n = 3

# C = sp.linalg.block_diag(np.zeros([n,n]), 1)

# P = np.random.rand(n,n)
# P = P.T @ P

# Q = np.random.rand(n,n)
# Q = Q.T @ Q

#########################################################################

selection = np.zeros(n)
selection[-1] = 1

const = 1
pencil = LinearMatrixPencil2(Q, P)

print("Pencil eigenvectors = ")
print(pencil.eigenvectors)

print("Pencil eigenvalues = ")
print(pencil.eigenvalues)

mu1, mu2, Z = pencil.solve_nonlinear(const)
num_sols = len(mu1)

print("Computed mu1's = ")
print(mu1)

print("Computed mu2's = ")
print(mu2)

print("Computed eigenvectors = ")
print(Z)

error_pencil = np.zeros([num_sols,num_sols])
error_eigenvalues = np.zeros(num_sols)
error_mu2 = np.zeros(num_sols)
error_kernel_image = np.zeros(num_sols)
error_boundaries = np.zeros(num_sols)
for k in range(num_sols):

    # L = pencil.eigenvalues[k] * pencil._A - pencil._B
    # L = pencil.value( mu1[k] / mu2[k] )
    L = mu1[k] * pencil._A - mu2[k] * pencil._B
    z = Z[:,k]

    print("Error pencil " + str(k+1) + " = " + str(np.linalg.norm(L @ z)))
    error_mu2[k] = mu2[k] - 0.5 * const * z @ P @ z
    error_kernel_image[k] = selection @ z - 1
    error_boundaries[k] = z @ Q @ z - 1

print("Error mu2 = " + str(error_mu2))
print("Error kernel image = " + str(error_kernel_image))
print("Error boundary = " + str(error_boundaries))