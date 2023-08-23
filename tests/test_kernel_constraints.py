import numpy as np
from functions import Kernel
from common import ret_basis, generate_monomial_list, kernel_constraints

n = 3
max_degree = 3
alpha, list = generate_monomial_list(n,max_degree)

initial_state = [ np.random.rand() for _ in range(n) ]
kernel = Kernel(*initial_state, degree = max_degree)
p = kernel.kernel_dim
print(kernel)

num_tests = 1000
error = 0
for _ in range(num_tests):
    z = kernel.function( np.random.rand(n) )
    error += np.linalg.norm(kernel_constraints(z, list))

print("Error = " + str(error))

# z = kernel.function( np.random.rand(n) )
# while kernel.is_in_kernel_space( z ):
#     z = kernel.function( np.random.rand(n) )
#     print("Hope I'm trapped in an infinite loop right now")