import numpy as np
from numpy.linalg import eigvals as eigs

from itertools import product
from common.math import MatrixPolynomial

shape1 = (2)
degree1 = 2

coefs1 = [ np.random.randn(shape1) for _ in range(degree1+1) ]
poly1 = MatrixPolynomial(coefs1)

shape2 = (2)
degree2 = 2

coefs2 = [ np.random.randn(shape2) for _ in range(degree2+1) ]
poly2 = MatrixPolynomial(coefs2)

num_updates = 1
poly = poly1
for _ in range(num_updates):

    shape1 = (2)
    degree1 = 2

    coefs1 = [ 10*np.ones(shape1) for _ in range(degree1+1) ]
    poly1.update(coefs1)
    print(poly1)

    shape2 = (2)
    degree2 = 2

    coefs2 = [ np.ones(shape2) for _ in range(degree2+1) ]
    poly2.update(coefs2)
    print(poly2)

    print( np.outer( poly1, poly2 ).shape )

    res = MatrixPolynomial.from_array( np.outer( poly1, poly2 )[0,0] )
    print(res)