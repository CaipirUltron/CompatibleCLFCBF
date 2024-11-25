import numpy as np
from numpy.linalg import eigvals as eigs

from numpy.polynomial import Polynomial as Poly
from common import MatrixPencil
from common import MatrixPolynomial as mp

shape = (2,2)

M = np.random.randn(*shape)
N = np.random.randn(*shape)
pencil = MatrixPencil( M, N )

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

    M = np.random.randn(*shape)
    N = np.random.randn(*shape)
    pencil.update(M = M, N = N)
    
    if pencil.type == 'regular':

        determinant, adjoint = pencil.inverse()

        ''' Test eigenvalues/eigenvectors (only for regular matrix pencils) '''
        for k, eig in enumerate(pencil.eigens):

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
        detDiag = np.array([[ determinant if j==i else Poly([0.0], symbol=pencil.symbol) for j in range(shape[1]) ] for i in range(shape[0]) ])
        adjPolyError = pencil @ adjoint - detDiag

        for index, _ in np.ndenumerate( pencil.coef[0] ):
            adjoint_error += sum( adjPolyError[index].coef )

    ''' Test nullspace (only for singular matrix pencils) '''
    if pencil.type == 'singular':

        nullPoly = pencil.nullspace()
        error_poly = pencil @ nullPoly
        for index, poly in np.ndenumerate(error_poly): null_error += np.linalg.norm( poly.coef )

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
