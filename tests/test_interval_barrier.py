import numpy as np
from numpy.polynomial import Polynomial as Poly
import matplotlib.pyplot as plt

from controllers import interval_barrier

from sympy import Poly as symPoly
from sympy.abc import x

p = Poly.fromroots([1,3,5,7])
p_sym = sum([ coef*(x**k) for k, coef in enumerate(p.coef) ])

hor_trans = 0.049
ver_trans = 0

p_sym = p_sym.subs({x:x-hor_trans})
p_sym += ver_trans
p_sym = symPoly(p_sym)
trans_coeffs = [ float(coef) for coef in p_sym.all_coeffs()[::-1] ]
p = Poly(trans_coeffs)

interval = [3.8, 4.2]

barrier = interval_barrier(p, interval)
print(f"Interval barrier = {barrier}")

l_min, l_max = -10, 10
l_array = np.arange(l_min,l_max,0.1)
p_array = [ p(l) for l in l_array ]

fig, ax = plt.subplots(figsize=(10.0, 5.0), layout="constrained")

ax.plot(interval, [0.0, 0.0], '*r')
ax.plot( [l_min, l_max], [ 0.0, 0.0 ], 'k' )       # horizontal axis
ax.plot(l_array, p_array)
ax.set_ylim(-10, +10)

plt.show()