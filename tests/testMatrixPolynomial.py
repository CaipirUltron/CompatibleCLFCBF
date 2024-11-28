import numpy as np
from common import MatrixPolynomial

degree1 = 2
shape1 = (2,2)

coefs1 = [ np.random.randn(*shape1) for _ in range(degree1+1) ]
poly1 = MatrixPolynomial(coefs1)

degree2 = 2
shape2 = (2,2)

coefs2 = [ np.random.randn(*shape2) for _ in range(degree2+1) ]
poly2 = MatrixPolynomial(coefs2)

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

    poly_res = poly2 @ poly1.T
    print(poly_res)

    res = MatrixPolynomial( coef=poly_res )

    print(res)