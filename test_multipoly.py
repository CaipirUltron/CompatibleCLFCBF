import operator
import numpy as np
import sympy as sym

from common import generate_monomials
from functions import MultiPoly as Poly

# ------------------------------------ Define kernel -----------------------------------
n = 2
powers, _ = generate_monomials(n, max_degree=[2,2])
print(f"Powers = {powers}")
p = len(powers)

data_type = "scalar"
# data_type = "vector"
# data_type = "matrix"

if data_type == "scalar": args = []
elif data_type == "vector": args = [2]
elif data_type == "matrix": args = [2,2]

coeffs1 = [ np.random.randn(*args) for _ in range(p) ]
coeffs2 = [ np.random.randn(*args) for _ in range(p) ]

p1 = Poly(kernel=powers, coeffs=coeffs1)
p2 = Poly(kernel=powers, coeffs=coeffs2)

print(f"p1 = {p1}")
print(f"p2 = {p2}")

p1.sort_kernel()

print("p1 after sorting = \n")
print(p1)

p1.filter()

print(f"p1 after filtering = \n")
print(p1)

zero_poly = Poly.zeros(kernel=powers)
print(f"Zero polynomial = \n{zero_poly}")

for k, p in enumerate(p1.polyder()):
    print(f"{k+1}-th derivative of p1 = \n{p}")

def test_op(op, N):
    ''' Performs N tests on operation op '''

    exponent = np.random.randint(0,10)

    if op != operator.pow:
        op_poly = op(p1, p2)
    else:
        op_poly = p1**exponent

    error = 0.0
    for _ in range(N):

        x = np.random.rand(n)

        if op != operator.pow:
            res_p1p2 = op(p1(x), p2(x))
        else:
            if p1.ndim != 2:
                res_p1p2 = p1(x)**exponent
            else:
                res_p1p2 = np.linalg.matrix_power(p1(x), exponent)

        res = op_poly(x)
        error += np.linalg.norm( res - res_p1p2 )

    print(f"Total error in {op.__name__} operation: {error}\n")

# -------------------------------- Run numeric tests -----------------------------------
N = 1000

print(f"Running {N} random numeric tests with {data_type}-valued polynomials.\n")
test_op(operator.add, N)
test_op(operator.sub, N)
test_op(operator.mul, N)
if data_type != "scalar": test_op(operator.matmul, N)
test_op(operator.pow, N)

# -------------------------------- Run symbolic tests -----------------------------------

# coeffs1 = [ sym.MatrixSymbol("U{}".format(k), *args) for k in range(p) ]
# coeffs2 = [ sym.MatrixSymbol("V{}".format(k), *args) for k in range(p) ]

# P1 = MultiPoly(kernel=kernel._powers, coeffs=coeffs1)
# P2 = MultiPoly(kernel=kernel._powers, coeffs=coeffs2)

# coeffs = [ sym.Symbol("u{}".format(k)) for k in range(p) ]
# p = MultiPoly(kernel=kernel._powers, coeffs=coeffs)

# sos_kernel = p.sos_kernel()
# sos_index_matrix = p.sos_index_matrix(sos_kernel)
# shape_matrix = p.shape_matrix(sos_kernel, sos_index_matrix)

# print( p )

# shape_fun = sym.lambdify( coeffs, shape_matrix )

# clf = KernelQuadratic(kernel=kernel, coefficients=np.random.randn(6,6), limits=limits, spacing=0.1 )
# print(f"CLF = \n{clf}")
# print(f"CLF params = \n{clf.shape_matrix}")

# poly, grad_poly, hessian_poly = clf.to_multipoly()
# print(f"CLF with equiv MultiPoly = {poly}")
# print(f"CLF gradient with equiv MultiPoly = {grad_poly}")
# print(f"CLF hessian with equiv MultiPoly = {hessian_poly}")