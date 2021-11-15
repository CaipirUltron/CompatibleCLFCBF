import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

def CLBF(x, y):
    c0, l0_x, l0_y = 1.0, 0.05, 0.05
    c1, l1_x, l1_y = 1.0, 1.0, 1.0
    x1, y1 = 3.0, 3.0
    return -c0 * np.exp(-(l0_x *x**2 + l0_y * y**2)) + c1 * np.exp(-(l1_x * (x-x1)**2 + l1_y * (y-y1)**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-6.0, 6.0, 0.02)
X, Y = np.meshgrid(x, y)
zs = np.array(CLBF(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()