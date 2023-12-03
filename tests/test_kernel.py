import time
import numpy as np
from functions import Kernel
from common import ret_basis, generate_monomial_list, kernel_constraints

num_tests = 100                     # Number of numerical tests

n = 2
max_degree = 2
alpha, terms_by_degree = generate_monomial_list(n,max_degree)

initial_state = [ np.random.rand() for _ in range(n) ]
kernel = Kernel(*initial_state, degree = max_degree)
p = kernel.kernel_dim
N_list = kernel.get_N_matrices()
A_list = kernel.get_A_matrices()

print("Initiating kernel tests.")
print(kernel)
time.sleep(3)

'''
Jacobian compliance test.
'''
print("\n Kernel jacobian compliance test.")
print("Checking if the Jacobian component matrices A are correct.")
print("Checking for " + str(num_tests) + " random samples.")
it = 0
total_error = 0.0
while it < num_tests:
    it += 1
    x = np.random.rand(n)
    m = kernel.function(x)

    Jm = kernel.jacobian(x)
    Jm_with_A = np.zeros([p,n])
    for k in range(len(A_list)):
        Jm_with_A[:,k] = A_list[k] @ m

    total_error += np.linalg.norm(Jm - Jm_with_A)
print("The total error for all samples is " + str(total_error))
if total_error < 1e-5:
    print("Kernel jacobian compliance test is a success!")
else:
    print("Kernel jacobian compliance test failure.")
time.sleep(3)

'''
Kernel null space test.
'''
print("\n Kernel null space test.")
print("Checking whether A.T N + N.T A returns the null matrix for all combinations of A and N.")
comb = 0
total_error = 0.0
for N in N_list:
    for A in A_list:
        comb += 1
        frobenius_norm = np.linalg.norm(A.T @ N + N.T @ A, ord='fro')
        total_error += frobenius_norm
        print("Norm of A.T N + N.T A for combination " + str(comb) + " = " + str(frobenius_norm))
print("The total error for all combinations is " + str(total_error))
if total_error < 1e-5:
    print("Kernel null space test is a success!")
else:
    print("Kernel null space test failure.")
time.sleep(3)

'''
Kernel constraint test.
'''
print("\n Kernel constraints test.")
print("The kernel_constraints() method should return a null vector if its input is a vector in the kernel image set.")
print("Checking for " + str(num_tests) + " random samples.")
it = 0
total_error = 0.0
while it < num_tests:
    it += 1 
    x = np.random.rand(n)
    m = kernel.function(x)
    F, matrix_constraints = kernel_constraints(m, terms_by_degree)
    total_error += np.linalg.norm(F)
print("The total error for all samples is " + str(total_error))
if total_error < 1e-5:
    print("Kernel constraints test is a success!")
else:
    print("Kernel constraints space test failure.")
time.sleep(3)
