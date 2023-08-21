'''
Test polynomial functions
'''
import numpy as np
import matplotlib.pyplot as plt

from common import *
from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelQuadratic

plt.rcParams['text.usetex'] = True

n, m = 2, 2
max_degree = 2

state = np.random.rand(n)
control = np.random.rand(m)

kernel = Kernel(*state, degree = max_degree)
kernel_dim = kernel.kernel_dim
print(kernel)

print("\n")

Proot = np.random.rand(kernel_dim, kernel_dim)
clf = KernelQuadratic(*state, kernel = kernel, coefficients = Proot.T @ Proot)
print(clf)

V = clf.function(state)
nablaV = clf.gradient(state)
HV = clf.hessian(state)

print("V = " + str(V))
print("nablaV = " + str(nablaV))
print("HessianV = " + str(HV))

print("\n")

F = np.random.rand(kernel_dim, kernel_dim)
def g(state):
    return np.eye(n)

system = ConservativeAffineSystem(initial_state=state, initial_control=control, kernel = kernel, F = F, g_method = g )
print(system)

sample_time = 0.001
system.set_control(control) 
system.actuate(sample_time)

Psol = clf.define_zeros( np.array([3.0, 4.0]) )

# kernel = clf.get_kernel()
# k_dim = clf.kernel_dim

# print("System dimension is "+str(n)+".")
# print("Kernel = " + str(kernel)+".")
# print("Kernel dimension is " + str(k_dim)+".")
# print("Therefore... I need at least " + str(k_dim-n) + " solutions.")

p = [3, 6, 10, 15, 21, 28,36]
sols = [3,9,22,45,81,133,204]

# plt.plot(p, sols, '--', color = 'red')
# plt.show()

# A_list = clf.get_A_matrices()
# N_list = clf.get_N_matrices()
# roundness = 5

# print(str(len(N_list)) + " solutions were found.")

# for k in range(len(N_list)):
#     N = N_list[k]

    # print(str(k+1)+'-th solution = \n')
    # print(str(np.round(N,roundness))+'\n')

    # print('Symmetry of N:')
    # print( np.linalg.norm(N - N.T) )
    # print("\n")

    # print('Skew-symmetry of A\'N:')
    # for Ai in A_list:
    #     print( np.linalg.norm(Ai.T @ N + N.T @ Ai) )
    #     print("\n")