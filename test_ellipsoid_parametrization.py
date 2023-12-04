import numpy as np
from controllers.equilibrium_algorithms import generate_point_grid

p = 6
Q = np.random.rand(p,p)
Q = Q @ Q.T

generate_point_grid(Q, 0.0)