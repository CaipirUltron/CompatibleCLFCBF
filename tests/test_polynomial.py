'''
Test polynomial functions
'''
import numpy as np
from functions import Kernel, PolynomialFunction
from common import *
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

n = 2
max_degree = 6

# P = np.random.rand(p,p)
# P = P.T @ P
# eigP , _= np.linalg.eig(P)
# P = P/np.max(eigP)

state = np.random.rand(n)
kernel = Kernel(*state, degree = max_degree)
kernel_dim = kernel._num_monomials
print(kernel)

print("\n")

Proot = np.random.rand(kernel_dim, kernel_dim)
clf = PolynomialFunction(*state, kernel = kernel, P = Proot.T @ Proot)
print(clf)

V = clf.function(state)
nablaV = clf.gradient(state)
HV = clf.hessian(state)

print("V = " + str(V))
print("nablaV = " + str(nablaV))
print("HessianV = " + str(HV))

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