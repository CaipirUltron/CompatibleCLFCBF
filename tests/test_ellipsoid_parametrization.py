import numpy as np
from controllers.equilibrium_algorithms import generate_point_grid

p = 6
Q = np.random.rand(p,p)
Q = Q @ Q.T

resolution = 0.8                # resolution for angular parameters. USE > 0.5, OTHERWISE MEMORY LEAK
points = generate_point_grid(Q, resolution)
print("Number of generated points = " + str(len(points)))

mQm_error = 0.0
for pt in points:
    mQm_error += ( pt.T @ Q @ pt - 1 )
print("Tested for " + str(len(points)) + " points, with total error = " + str(mQm_error))