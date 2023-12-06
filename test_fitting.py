import matplotlib.pyplot as plt
from functions import Kernel, KernelLyapunov, KernelBarrier

initial_state = [3.2, 3.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
max_degree = 2
kernel = Kernel(*initial_state, degree = max_degree)
kern_dim = kernel.kernel_dim
print(kernel)
print("Dimension of kappa space = " + str(len(kernel.get_N_matrices())))

# ----------------------------------------------------- Define CBF ---------------------------------------------------------
boundary_points = [ [-4.0, 0.0], [-4.0, -1.0], [-2.0, 0.5], [2.0, 0.5], [4.0, -1.0], [4.0, 0.0], [0.0, 1.0], [0.0, -0.5] ]   # (sad smile)
# boundary_points = [ [-4.0, 0.0], [-4.0, 1.0], [-2.0, -0.5], [2.0, -0.5], [4.0, 1.0], [4.0, 0.0] ]                          # (bell shaped)

cbf = KernelBarrier(*initial_state, kernel=kernel, boundary_points=boundary_points)

points = []
points.append({ "point": [2.0, 2.0], "gradient": [1.0, 1.0] })
points.append({ "point": [2.0, -2.0], "gradient": [1.0, -1.0] })
points.append({ "point": [-2.0, -2.0], "gradient": [-1.0, -1.0] })
points.append({ "point": [-2.0, 2.0], "gradient": [-1.0, 1.0] })
# points.append({ "point": [0.0, 2.0], "gradient": [0.0, 1.0] })
# points.append({ "point": [0.0, -2.0], "gradient": [0.0, -1.0] })

cbf.fit(points)
print(cbf.matrix_coefs)

ax = cbf.plot(axeslim = [-10, 10, -10, 10])
for pt in points:
    ax.plot(pt["point"][0], pt["point"][1], "go")

plt.show()