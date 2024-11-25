import numpy as np
from numpy.linalg import eigvals as eigs

from itertools import product
from common.math import MatrixPolynomial as Poly

degree1 = 2
shape1 = (2,2)

coefs1 = [ np.random.randn(*shape1) for _ in range(degree1+1) ]
poly1 = Poly(coefs1)

degree2 = 2
shape2 = (2,2)

coefs2 = [ np.random.randn(*shape2) for _ in range(degree2+1) ]
poly2 = Poly(coefs2)

num_updates = 1
poly = poly1
for _ in range(num_updates):

    degree1 = 2
    coefs1 = [ np.random.randn(*shape1) for _ in range(degree1+1) ]
    poly1.update(coefs1)
    print(poly1)

    degree2 = 2
    coefs2 = [ np.random.randn(*shape2) for _ in range(degree2+1) ]
    poly2.update(coefs2)
    print(poly2)
    print(poly2.T)
    
    # poly_res = Poly.outer( poly1, poly2 )
    # poly_res = poly2[0,1]
    poly_res = poly1.T @ poly2

    res = Poly.from_array( poly_res )

    print(res)