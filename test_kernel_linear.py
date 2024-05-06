import numpy as np
import sympy as sym

import matplotlib.pyplot as plt
from functions import Kernel, KernelLinear, MultiPoly
from common import generate_monomial_symbols

np.set_printoptions(precision=3, suppress=True)

# ------------------------------------ Define kernel -----------------------------------
n = 2
d = [2,3]
kernel = Kernel(dim=n, degree=3)

monomials=[(0,0),(1,0),(0,1),(2,0),(0,2),(2,3)]
# kernel = Kernel(dim=n, monomials=monomials)

p = kernel._num_monomials
print(kernel)

coeffs = [ np.random.rand() for _ in range(p) ]
poly = KernelLinear(kernel=kernel, coeffs=coeffs)

print(f"Kernel monomials = { kernel._monomials }" )
sym_coeffs = [ sym.Symbol(f"c{k}") for k in range(p) ]
print(f"Coefficients = {sym_coeffs}")

sos_kernel = Kernel(dim=n, monomials=poly._sos_monomials)
print(f"SOS monomials = { sos_kernel._monomials }" )
sym_shape_matrix = poly.get_sos_shape(sym_coeffs)
print(f"SOS shape matrix = \n{np.array(sym_shape_matrix)}")

num_tests = 100

total_error = 0.0
for i in range(num_tests):

    x = np.random.rand(n)
    coeffs = [ np.random.rand() for _ in range(p) ]

    poly.set_params(coeffs = coeffs)
    f1 = poly.function(x)

    s = sos_kernel.function(x)
    M = np.array(poly.get_sos_shape(coeffs))
    f2 = s.T @ M @ s

    error = np.abs(f1 - f2)
    total_error += error
    # print(f"Error at {i+1}-iteration = {error}")

print(f"Final error = {total_error}")