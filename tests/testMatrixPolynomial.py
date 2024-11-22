import numpy as np
from numpy.linalg import eigvals as eigs

from itertools import product
from controllers import MatrixPolynomial

shape = (2,3)
degree = 2

coefs = [ np.random.randn(*shape) for _ in range(degree+1) ]
poly_matrix = MatrixPolynomial(*coefs)
print(poly_matrix)
print(f"Coefficients of P({poly_matrix.symbol}):")
for k, c in enumerate(poly_matrix.coef):
    print(f"P{k} = \n{c}")

degree = 4

# update_method = 'args'
update_method = 'kargs'

if update_method == 'args':
    coefs = [ np.random.randn(*shape) for _ in range(degree+1) ]
    print("New produced coefficients:")
    for k, c in enumerate(coefs):
        print(f"P{k} = \n{c}")
    poly_matrix.set(*coefs)

if update_method == 'kargs':
    # coefs = { str(i):np.random.randn(*shape) for i in range(degree+1) }
    coefs = { str(2):np.random.randn(*shape) }
    print("New produced coefficients:")
    for k, c in coefs.items():
        print(f"P{k} = \n{c}")
    poly_matrix.set(**coefs)

print(f"Coefficients of P({poly_matrix.symbol}) after update:")
for k, c in enumerate(poly_matrix.coef):
    print(f"P{k} = \n{c}")
