import numpy as np
import matplotlib.pyplot as plt
from functions import Kernel, KernelQuadratic

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

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
base_level = 16.0
points = []
points += [{ "point": [ 0.0,  0.0], "level": 0.0 }]
points += [{ "point": [ 3.0,  3.0], "level": base_level, "gradient": [ 1.0,  1.0] }]
points += [{ "point": [-3.0,  3.0], "level": base_level, "gradient": [-1.0,  1.0] }]
points += [{ "point": [ 0.0,  5.0],                      "gradient": [ 0.0,  1.0], "curvature": -1.6 }]

function = KernelQuadratic(*initial_state, kernel=kernel, points=points)
# function.plot_level(level = 20, axeslim = [-10, 10, -10, 10])
# plt.show()

A_list = kernel.get_A_matrices()
Q = function.matrix_coefs

'''
Compares the computation of the function hessian matrix directly by diff
and by means of the A matrices
'''
num_tests = 100
errors = []
for k in range(num_tests):

    x = np.random.rand(n)
    m = kernel.function(x)

    Hessian1 = function.hessian(x)
    Hessian2 = np.zeros([n,n])
    for i in range(n):
        Ai = A_list[i]
        for j in range(n):
            Aj = A_list[j]
            Hessian2[i,j] = m.T @ Ai.T @ Q @ Aj @ m + m.T @ Q @ Ai @ Aj @ m 

    errors.append( np.linalg.norm( Hessian1 - Hessian2 ) )

print("Error after " + str(num_tests) + " tests = " + str(np.sum(errors)) )