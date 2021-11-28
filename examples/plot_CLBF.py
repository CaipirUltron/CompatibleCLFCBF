import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from compatible_clf_cbf.dynamic_systems import Gaussian, CLBF

# Create CLBF function with goal and obstacle list.
x0 = [0.0, 0.0]
goal = Gaussian(init_value=x0, constant = 1.0, mean = [0.0, 0.0], shape = 0.05*np.eye(2))
obstacle1 = Gaussian(init_value=x0, constant = 1.0, mean = [3.0, 3.0], shape = np.eye(2))
obstacle2 = Gaussian(init_value=x0, constant = 1.0, mean = [-3.0, -3.0], shape = np.eye(2))
clbf = CLBF( init_value=x0, goal=goal, obstacles=[obstacle1, obstacle2] )

# Create figure and meshgrids for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-6.0, 6.0, 0.1)
X, Y = np.meshgrid(x, y)

# Evaluate the CLBF function on the meshgrid
zs = 0*X
for i in range(len(X)):
    for j in range(len(Y)):
        state = [ X[i,j], Y[i,j] ]
        zs[i,j] = clbf.evaluate(state)
Z = zs.reshape(X.shape)

# Plot CLBF
ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()