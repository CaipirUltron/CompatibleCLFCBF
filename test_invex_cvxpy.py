import itertools
import warnings

import numpy as np
import matplotlib.pyplot as plt

from functions import Kernel
from common import hessian_2Dquadratic, PSD_closest, NSD_closest, timeit

# np.set_printoptions(precision=3, suppress=True)
limits = 3*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test Invex CLFs")
ax.set_aspect('equal', adjustable='box')

ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

# ------------------------------------ Define kernel and CLF -----------------------------------
n, d = 2, 2
kernel = Kernel(dim=n, degree=d)
print(kernel)
kernel_dim = kernel._num_monomials
q = kernel.dim_det_kernel
p = kernel._num_monomials

print(f"D(N) = \n{kernel.Dfun_symbolic}")
for i, j in itertools.product( range(n), range(p) ):
    der = np.array(kernel.Dfun_symbolic_derivatives[i][j])
    print(f"D'{i}{j}(N) = \n{der}")

N = np.random.randn(n,p)
list = kernel.Dderivatives(N)
print(list[0][5])

#---------------------------------- Function for invexification -----------------------------------
# @timeit
def find_invex(Ninit: np.ndarray):
    '''
    Returns an N matrix that produces an invex function k(x).T N.T P N K(x) on the given kernel k(x).
    PROBLEM: find N such that shape_fun(N) >> 0
    '''
    pass