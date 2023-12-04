import numpy as np
from controllers import LinearMatrixPencil
import cvxpy as cp

dim = 6
num_tests = 100

Q = np.random.rand(dim, dim)
L = np.random.rand(dim, dim)

# var = cp.Variable((dim, dim))
# param = cp.Parameter((dim, dim))

# objective = cp.Minimize( cp.norm( var - param , "fro") )
# constraint = [ var + var.T >> 0 ]
# problem = cp.Problem(objective, constraint)

# param.value = L
# problem.solve()
# L = var.value

'''
Test for pencil eigenpairs.
'''
print("Pencil eigenpairs test.")
print("Checking whether the solutions (l Q - L) z = 0 are correct for linear matrix pencil (Q,L).")

pencil = LinearMatrixPencil(Q @ Q.T, L @ L.T)

it = 0
pencil_matrix_error = 0.0
pencil_det_eig_error = 0.0
pencil_det_lambda_error = 0.0
pencil_eigen_error = 0.0
while it < num_tests:
    it += 1

    Q = np.random.rand(dim, dim)
    L = np.random.rand(dim, dim)
    pencil.set_pencil(A = Q @ Q.T, B = L @ L.T)

    num_eigen = len(pencil.eigenvalues)
    for k in range(num_eigen):
        lambda1 = pencil.lambda1[k]
        lambda2 = pencil.lambda2[k]
        eig = pencil.eigenvalues[k]
        z = pencil.eigenvectors[k]

        pencil_matrix_eig = pencil.value(eig)
        pencil_matrix_lambda = pencil.value_double(lambda1, lambda2)

        pencil_matrix_error += np.linalg.norm( pencil_matrix_eig - pencil_matrix_lambda , 'fro')

        pencil_matrix_det_eig = np.linalg.det( pencil_matrix_eig )
        pencil_matrix_det_lambda = np.linalg.det( pencil_matrix_lambda )

        pencil_det_eig_error += pencil_matrix_det_eig
        pencil_det_lambda_error += pencil_matrix_det_lambda

        pencil_eigen_error += np.linalg.norm(pencil_matrix_lambda @ z)

print("Exit after testing with " + str(it) + " random samples. \n")
print("Pencil matrix error (diff. btw computing with lambdas or ratios) = " + str(pencil_matrix_error))
print("Pencil determinant error (on ratio computed pencil) = " + str(pencil_det_eig_error))
print("Pencil determinant error (on lambda computed pencil) = " + str(pencil_det_lambda_error))
print("Pencil eigenpair error (residue of the gen. eigenproblem) = " + str(pencil_eigen_error))
