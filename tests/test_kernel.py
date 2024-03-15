import time
import numpy as np
from scipy.linalg import null_space
from functions import Kernel
from common import ret_basis, generate_monomial_list, kernel_constraints

num_tests = 100                     # Number of numerical tests

n = 2
max_degree = 3
alpha, terms_by_degree = generate_monomial_list(n,max_degree)

initial_state = [ np.random.rand() for _ in range(n) ]
kernel = Kernel(*initial_state, degree = max_degree)
p = kernel.kernel_dim
N_list = kernel.get_N_matrices()
A_list = kernel.get_A_matrices()
r = len(N_list)

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
    print("Kernel constraints test failure.")
time.sleep(3)


'''
Kernel kappa test.
'''
print("\n Kernel kappa test.")
print("Checks whether the kappas actually can be used to compute the null space vectors for the Jacobian transpose.")
print("Checking for " + str(num_tests) + " random samples.")
it = 0
errors = []
while it < num_tests:
    it += 1
    x = np.random.rand(n)
    m = kernel.function(x)
    Jm = kernel.jacobian(x)

    coeff_matrix = np.array([ (N @ m).tolist() for N in N_list ]).T
    nulls = null_space(Jm.T)
    for i in range(nulls.shape[1]):
        null = nulls[:,i]
        sol = np.linalg.lstsq(coeff_matrix, null, rcond=None)
        kappa = sol[0]

        null_vec_kappa = np.zeros(p)
        for i in range(r):
            null_vec_kappa += kappa[i] * N_list[i] @ m

        errors.append( np.linalg.norm( null_vec_kappa - null ) )

total_error = np.sum(errors)
print("The total error for all samples is " + str(total_error))
if total_error < 1e-5:    
    print("Kernel kappa test is a success!")
else:
    print("Kernel kappa test failure.")
time.sleep(3)