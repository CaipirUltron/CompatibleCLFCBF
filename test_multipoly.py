import operator
import numpy as np
import sympy as sym

from numpy.polynomial import Polynomial as npPoly
from numpy.polynomial.polynomial import polyint, polyder

from common import generate_monomials, rot2D, kernel_quadratic
from functions import MultiPoly
from functions import Poly as myPoly
from functions import Kernel, KernelQuadratic, KernelLyapunov, KernelBarrier

# ------------------------------------ Define kernel -----------------------------------
n = 2
powers, _ = generate_monomials(n, max_degree=4)
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

p1 = MultiPoly(kernel=powers, coeffs=coeffs1)
p2 = MultiPoly(kernel=powers, coeffs=coeffs2)

print(f"p1 = {p1}")
print(f"p2 = {p2}")

p1.sort_kernel()

print("p1 after sorting = \n")
print(p1)

p1.filter()

print(f"p1 after filtering = \n")
print(p1)

zero_poly = MultiPoly.zeros(kernel=powers)
print(f"Zero polynomial = \n{zero_poly}")

p1.poly_diff()
for k, p in enumerate(p1._poly_diff):
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

if data_type != "scalar": 
    test_op(operator.matmul, N)

if data_type in ("scalar", "matrix"):
    test_op(operator.pow, N)

if data_type == 'vector':
    p_comp = p1(p2)

    t = np.random.randn(n)
    R = rot2D(np.deg2rad( np.random.randint(-180, 180) ))
    p1_transform = p1.frame_transform(t, R)

    comp_error = 0.0
    rigid_error = 0.0

    for _ in range(N):

        x = np.random.rand(n)

        res_pcomp1 = p1(p2(x))
        res_pcomp2 = p_comp(x)
        comp_error += np.linalg.norm( res_pcomp1 - res_pcomp2 )

        x_new = R @ x - t
        res_ptransf1 = p1(x_new)
        res_ptransf2 = p1_transform(x)
        rigid_error += np.linalg.norm( res_ptransf1 - res_ptransf2 )

    print(f"Total error in composition operation: {comp_error}\n")
    print(f"Total error in rigid-body transformation: {rigid_error}\n")

    poly_outer = MultiPoly.outer( p1, p2 )
    print(poly_outer)

if data_type == "scalar":

    shape = p1.shape_matrix()
    p1.poly_grad()
    p1.poly_hess()

    print(f"gradient = {p1._poly_grad}")
    print(f"hessian = {p1._poly_hess}")

    gamma = myPoly([0.0, 1.0])
    intgamma = myPoly( polyint(gamma.coef) )

    newp = gamma(p1)
    print(newp)

    kernel = Kernel(dim=n, degree=2)
    kernel_dim = kernel._num_monomials

    clf_center = [0.0, -3.0]
    clf_eig = np.array([ 1.0, 1.0 ])
    clf_angle = np.deg2rad(-45)
    Pquadratic = kernel_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim)

    clf_poly = KernelLyapunov(kernel=kernel, P=Pquadratic).to_multipoly()

    inverse_gamma_error = 0.0
    for i in range(N):
        x = np.random.rand(n)
        V, gradV, Hv = clf_poly.inverse_gamma_transform( x, gamma )
        inverse_gamma_error += np.abs( clf_poly(x) - intgamma( V ) )

    print(f"Inverse gamma error = {inverse_gamma_error}")


p = npPoly([1, 2, 1])
my_p = MultiPoly.from_nppoly(p)

print(f"Kernel = {my_p.kernel}")
print(f"Coeffs = {my_p.coeffs}")

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