import numpy as np
import matplotlib.pyplot as plt

from common import rgb
from functions import Kernel, KernelBarrier

initial_state = [3.2, 3.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
max_degree = 2
kernel = Kernel(*initial_state, degree = max_degree)
kern_dim = kernel.kernel_dim
print(kernel)
print("Kernel dimension = " + str(kernel.kernel_dim))

# ----------------------------------------------------- Define CBF ---------------------------------------------------------
boundary_points = [ [-4.0, 0.0], [-4.0, -1.0], [-2.0, 0.5], [2.0, 0.5], [4.0, -1.0], [4.0, 0.0], [0.0, 1.0], [0.0, -0.5] ]   # (sad smile)
# boundary_points = [ [-4.0, 0.0], [-4.0, 1.0], [-2.0, -0.5], [2.0, -0.5], [4.0, 1.0], [4.0, 0.0] ]                          # (bell shaped)

points = []
points += [ { "point": [ 2.0,  2.0], "level": 0.0, "gradient": [ 1.0,  1.0] } ]
points += [ { "point": [ 2.0, -2.0], "level": 0.0, "gradient": [ 1.0, -1.0] } ]
points += [ { "point": [-2.0, -2.0], "level": 0.0, "gradient": [-1.0, -1.0] } ]
points += [ { "point": [-2.0,  2.0], "level": 0.0, "gradient": [-1.0,  1.0] } ]
points += [ { "point": [ 0.0,  2.0], "level": 0.0, "gradient": [ 0.0,  1.0] } ]
# points += [ { "point": [ 0.0, -2.0], "level": 0.0, "gradient": [ 0.0, -1.0] } ]
# points += [ { "point": [ 2.0,  0.0], "level": 0.0, "gradient": [ 1.0,  0.0] } ]
# points += [ { "point": [-2.0,  0.0], "level": 0.0, "gradient": [-1.0,  0.0] } ]
displacement = 0*np.array([1,1])
for pt in points:
    # pt.pop("gradient")
    pt["point"] = ( np.array(pt["point"]) + displacement ).tolist()

cbf = KernelBarrier(*initial_state, kernel=kernel, points=points)
# cbf = KernelBarrier(*initial_state, kernel=kernel, boundary_points=boundary_points)

# cbf.fit(points)
# cbf.define_boundary(boundary_points)

ax = cbf.plot(axeslim = [-10, 10, -10, 10])
for pt in cbf.point_list:

    plt.plot(pt["point"][0], pt["point"][1], 'go')

    if "gradient" in pt.keys():
        if "curvature" in pt.keys():
            color = rgb(-10.0, 10.0, pt["curvature"])
            plt.quiver(pt["point"][0], pt["point"][1], pt["gradient"][0], pt["gradient"][1], color=color)
        else:
            plt.quiver(pt["point"][0], pt["point"][1], pt["gradient"][0], pt["gradient"][1])

plt.show()