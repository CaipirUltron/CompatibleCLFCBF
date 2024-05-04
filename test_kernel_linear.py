import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from functions import Kernel, KernelLinear
from common import lyap, create_quadratic, rot2D, symmetric_basis

np.set_printoptions(precision=3, suppress=True)
limits = 3*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test Invex CLFs")
ax.set_aspect('equal', adjustable='box')

ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

# ------------------------------------ Define kernel -----------------------------------
n = 2
d = [2,3]
kernel = Kernel(dim=n, degree=d)
print(kernel)
p = kernel._num_monomials

poly = KernelLinear(kernel=kernel, coefficients=[ np.random.rand(4,4) for _ in range(p) ])
print(poly._coefficients)