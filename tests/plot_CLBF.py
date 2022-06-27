import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from compatible_clf_cbf.dynamic_systems import Gaussian, CLBF

# Create CLBF function with goal and obstacle list.
x0 = [0.0, 0.0]
goal = Gaussian(init_value=x0, constant = 1.0, mean = [0.0, 0.0], shape = 0.1*np.eye(2))
obstacle1 = Gaussian(init_value=x0, constant = 2.0, mean = [3.0, 3.0], shape = np.eye(2))
obstacle2 = Gaussian(init_value=x0, constant = 1.0, mean = [-3.0, -3.0], shape = np.eye(2))
clbf = CLBF( init_value=x0, goal=goal, obstacles=[obstacle1] )

# Create random population
N = 100
grid_range = [-12, 12]
pop_values = np.zeros(N)
seed = np.random.rand(2,N)
population = grid_range[0]*(1-seed) + grid_range[1]*seed

gamma = 0.01
num_iter = 2000
for i in range(num_iter):
    for k in range(N):
        clbf.set_value(population[:,k])
        clbf.function()
        clbf.gradient()
        nabla_clbf = clbf.get_gradient()
        population[:,k] = population[:,k] - gamma*nabla_clbf/np.linalg.norm(nabla_clbf)
        pop_values[k] = clbf.get_fvalue()

# Create figure and meshgrids for plotting
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-6, 6, 0.1)
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
ax.scatter(population[0,:], population[1,:], pop_values, marker='o', color = 'k', alpha=1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()