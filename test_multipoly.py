import operator
import numpy as np
import sympy as sym

from functions import Kernel, KernelLinear, MultiPoly

np.set_printoptions(precision=3, suppress=True)

# ------------------------------------ Define kernel -----------------------------------
n = 2
d = [2,3]
kernel = Kernel(dim=n, degree=4)

p = kernel._num_monomials
print(kernel)

# data_type = "scalar"
# data_type = "vector"
data_type = "matrix"

size = 2
if data_type == "scalar": args = []
elif data_type == "vector": args = [size]
elif data_type == "matrix": args = [size, size]

coeffs1 = [ np.random.randn(size) for _ in range(p) ]
coeffs2 = [ np.random.randn() for _ in range(p) ]

# coeffs1 = [ np.random.randint(low=-4, high=4, size=args) for _ in range(p) ]
# coeffs2 = [ np.random.randint(low=-4, high=4, size=args) for _ in range(p) ]

# coeffs1 = [ np.eye(size) for _ in range(p) ]
# coeffs2 = [ np.eye(size) for _ in range(p) ]

p1 = MultiPoly(kernel=kernel._powers, coeffs=coeffs1)
p2 = MultiPoly(kernel=kernel._powers, coeffs=coeffs2)

print( p1.sos_kernel() )

linear1 = KernelLinear.from_poly(p1)
linear2 = KernelLinear.from_poly(p2)

def test_op(op, N):
    ''' Performs N tests on operation op '''

    op_poly = op(p1, p2)
    linear_res = KernelLinear.from_poly( op_poly )

    error = 0.0
    for _ in range(N):

        x = np.random.rand(n)

        res_p1 = linear1.function(x)
        res_p2 = linear2.function(x)

        res_p1p2 = op(res_p1, res_p2)
        res = linear_res.function(x)
        error += np.linalg.norm( res - res_p1p2 )

    print(f"Total error in {op.__name__} operation: {error}\n")

# -------------------------------- Run numeric tests -----------------------------------
N = 100

# print(f"Running {N} random numeric tests with {data_type}-valued polynomials.\n")
# test_op(operator.add, N)
# test_op(operator.sub, N)
# test_op(operator.mul, N)
# if data_type != "scalar": test_op(operator.matmul, N)

# print( p1.determinant() )

# -------------------------------- Run symbolic tests -----------------------------------

coeffs1 = [ sym.MatrixSymbol("U{}".format(k), *args) for k in range(p) ]
coeffs2 = [ sym.MatrixSymbol("V{}".format(k), *args) for k in range(p) ]

P1 = MultiPoly(kernel=kernel._powers, coeffs=coeffs1)
P2 = MultiPoly(kernel=kernel._powers, coeffs=coeffs2)

coeffs = [ sym.Symbol("u{}".format(k)) for k in range(p) ]
p = MultiPoly(kernel=kernel._powers, coeffs=coeffs)

sos_kernel = p.sos_kernel()
sos_index_matrix = p.sos_index_matrix(sos_kernel)
shape_matrix = p.shape_matrix(sos_kernel, sos_index_matrix)

print( p )

shape_fun = sym.lambdify( coeffs, shape_matrix )