import numpy as np
import sympy as sym

import matplotlib.pyplot as plt
from functions import Kernel, KernelLinear, MultiPoly
from common import generate_monomial_symbols

np.set_printoptions(precision=3, suppress=True)

# ------------------------------------ Define kernel -----------------------------------
n = 2
d = [2,3]
kernel = Kernel(dim=n, degree=2)

# monomials=[(0,0),(1,0),(0,1),(2,0),(0,2),(2,3)]
# kernel = Kernel(dim=n, monomials=monomials)

p = kernel._num_monomials
print(kernel)

coeffs1 = [ sym.Symbol(f"u{k}") for k in range(p) ]
coeffs2 = [ sym.Symbol(f"v{k}") for k in range(p) ]

poly1 = MultiPoly(kernel=kernel._powers, coeffs=coeffs1)
poly2 = MultiPoly(kernel=kernel._powers, coeffs=coeffs2)

print(f"Coffs1 = {poly1.coeffs}")
print(f"Coffs2 = {poly2.coeffs}")

sum_poly = poly1 + poly2
sub_poly1 = poly1 - poly2
sub_poly2 = poly2 - poly1
mul_poly = poly1 * poly2

print(f"Monomials = {sum_poly.kernel}")
print(f"Coffs1 + Coffs2 = {sum_poly.coeffs}")
print(f"Coffs1 - Coffs2 = {sub_poly1.coeffs}")
print(f"Coffs2 - Coffs1 = {sub_poly2.coeffs}")

print(f"Mons of p1(x) p2(x) = {mul_poly.kernel}")

print(f"Coeffs of p1(x) p2(x) =")
for coeff in mul_poly.coeffs:
    print(coeff)