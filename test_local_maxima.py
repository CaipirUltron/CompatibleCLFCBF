import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial as Poly
from scipy.optimize import minimize

size = 5.0
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(size, size), layout="constrained")
fig.suptitle('Test local maxima')
ax.set_aspect('auto')

# poly = -0.1*Poly([2,-2, 1])*Poly([-5, 1])*Poly([-6, 1])
poly = -0.1*Poly([-2, 1])*Poly([-5, 1])*Poly([-6, 1])

xmin, xmax = -10, +10
x_arr = np.arange(-10, 10, 0.1)
p_arr = [ poly(x) for x in x_arr ]

ax.plot( x_arr, p_arr )

x0 = 4.0

constr = [ {"type": "ineq", "fun": lambda x: poly(x)} ]
sol = minimize( fun=lambda var: 0.0, x0=x0, constraints=constr, method='SLSQP', options={"disp": True, "maxiter": 1000} )

print(f"Sol. found = {sol.x}")

ax.plot( sol.x, poly(sol.x), 'o' )

ax.set_xbound(xmin, xmax)
ax.set_ybound(-6, 3)
ax.grid()

plt.show()