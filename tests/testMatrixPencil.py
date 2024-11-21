import numpy as np
from numpy.linalg import eigvals as eigs

from itertools import product
from controllers import MatrixPencil

n, p = 2, 2

M = np.random.randn(n,p)
N = np.random.randn(n,p)
pencil = MatrixPencil(M, N)

print("Starting MatrixPencil unit tests.")

det_error = 0.0
det_ratio_error = 0.0
l_error = 0.0
r_error = 0.0
null_error = 0.0
symb_det_error = 0.0
adjoint_error = 0.0

numTests = 1000
for it in range(numTests):

    M = np.random.randn(n,p)
    N = np.random.randn(n,p)
    
    pencil.set(M=M, N=N)
    pencilPoly = pencil.to_poly_array()
    
    if pencil.type == 'regular':

        determinant, adjoint = pencil.inverse()

        ''' Test eigenvalues/eigenvectors (only for regular matrix pencils) '''
        eigens = pencil.get_eigen()
        for k, eig in enumerate(eigens):

            Pvalue = pencil(eig.alpha, eig.beta)
            PvalueRatio = pencil(eig.eigenvalue)
            zRight = eig.rightEigenvectors
            zLeft = eig.leftEigenvectors

            det_error += np.linalg.det( Pvalue )
            det_ratio_error += np.linalg.det( PvalueRatio )
            r_error += np.linalg.norm( Pvalue @ zRight )
            l_error += np.linalg.norm( Pvalue.T @ zLeft )
            symb_det_error += determinant(eig.eigenvalue)

        ''' Test pencil inverse '''
        detDiag = np.array([[ determinant if j==i else 0.0 for j in range(n) ] for i in range(n) ])
        adjPolyError = pencilPoly @ adjoint - detDiag
        for (i,j) in product(range(pencil.shape[0]), range(pencil.shape[1])):
            adjoint_error += sum(adjPolyError[i,j].coef)

    ''' Test nullspace (only for singular matrix pencils) '''
    if pencil.type == 'singular':

        nullPoly = pencil.nullspace()
        error_poly = pencilPoly @ nullPoly
        for index, poly in np.ndenumerate(error_poly):
            null_error += np.linalg.norm( poly.coef )

print(f"Exit after testing with {it+1} random samples. \n")

if pencil.type == 'regular':
    print(f"Pencil determinant error = {det_error}")
    print(f"Pencil determinant error (on ratio computed pencil) = {det_ratio_error}")
    print(f"Pencil left eigenvector error = {l_error}")
    print(f"Pencil right eigenvector error = {r_error}")
    print(f"Pencil symbolic determinant error = {symb_det_error}")
    print(f"Pencil adjoint error = {adjoint_error}")

if pencil.type == 'singular':
    print(f"Pencil nullspace error = {null_error}")
