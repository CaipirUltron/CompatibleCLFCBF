import numpy as np
import matplotlib.pyplot as plt

from functions import Kernel, KernelQuadratic

limits = 12*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test level sets")
ax.set_aspect('equal', adjustable='box')

xmin, xmax, ymin, ymax = limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=2, degree=2)
kernel_dim = kernel._num_monomials

P = 0.1*np.random.randn(kernel_dim, kernel_dim)
P = P.T @ P
fun = KernelQuadratic(kernel=kernel, constant = 0.5, coefficients = P, limits = limits, spacing = 0.1, color = 'b')

print(fun.matrix_coefs)

fun.plot_levels(ax=ax, levels = [0.0])
plt.show()