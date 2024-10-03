import operator
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from common import generate_monomials
from functions.multipoly import MultiPoly as Poly

np.set_printoptions(precision=3, suppress=True)
limits = 14*np.array((-1,1,-1,1))

# fig = plt.figure(constrained_layout=True)
# ax = fig.add_subplot(111)
# ax.set_title("Hyperbolic Expansion")
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlim(limits[0], limits[1])
# ax.set_ylim(limits[2], limits[3])

# ------------------------------------ Define kernel -----------------------------------
n = 2
powers = generate_monomials(n, max_degree=[2,2])
print(f"Powers = {powers}")
p = len(powers)

data_type = "scalar"
# data_type = "vector"
# data_type = "matrix"

size = 2
if data_type == "scalar": args = []
elif data_type == "vector": args = [size]
elif data_type == "matrix": args = [size, size]

coeffs1 = [ np.random.randn(*args) for _ in range(p) ]
coeffs2 = [ np.random.randn(*args) for _ in range(p) ]

# coeffs1 = [ np.random.randint(low=-4, high=4, size=args) for _ in range(p) ]
# coeffs2 = [ np.random.randint(low=-4, high=4, size=args) for _ in range(p) ]

# coeffs1 = [ np.eye(size) for _ in range(p) ]
# coeffs2 = [ np.eye(size) for _ in range(p) ]

powers=[(1,0),(0,0),(2,1),(2,2),(0,1),(1,2),(1,1),(0,2),(2,0)]
p1 = Poly(kernel=powers, coeffs= [ k for k in range(9) ])
print(f"p1 = \n{p1}")

p1.sort_kernel()
print(f"p1 after sorting = \n{p1}")

p1.filter()
print(f"p1 after filtering = \n{p1}")

p1 = Poly(kernel=powers, coeffs=coeffs1)
p2 = Poly(kernel=powers, coeffs=coeffs2)

zero_poly = Poly.zeros(kernel=powers)
print(zero_poly)

# print(f"p1 = \n{p1}")
# print(f"p2 = \n{p2}")

# for k, p in enumerate(p1.polyder()):
#     print(f"{k+1}-th derivative of p1 = \n{p}")

def test_op(op, N):
    ''' Performs N tests on operation op '''

    op_poly = op(p1, p2)

    error = 0.0
    for _ in range(N):

        x = np.random.rand(n)

        res_p1p2 = op(p1(x), p2(x))
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