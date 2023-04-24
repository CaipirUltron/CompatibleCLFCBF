'''
Test polynomial functions
'''
import numpy as np
from functions import PolynomialFunction
from common import *
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

n = 2
max_degree = 8

# P = np.random.rand(p,p)
# P = P.T @ P
# eigP , _= np.linalg.eig(P)
# P = P/np.max(eigP)

state = np.random.rand(n)
clf = PolynomialFunction(*state, degree = max_degree)
kernel = clf.get_kernel()
k_dim = len(kernel)

print("System dimension is "+str(n)+".")
print("Kernel = " + str(kernel)+".")
print("Kernel dimension is " + str(k_dim)+".")
print("Therefore... I need at least " + str(k_dim-n) + " solutions.")
# print( generate_monomial_list(n, max_degree) )

A_list = clf.get_A_matrices()
N_list = clf.get_N_matrices()
roundness = 5

print(str(len(N_list)) + " solutions were found.")

p = [3, 6, 10, 15, 21, 28,36]
sols = [3,9,22,45,81,133,204]

# plt.plot(p, sols, '--', color = 'red')
# plt.show()

for k in range(len(N_list)):
    N = N_list[k]

    # print(str(k+1)+'-th solution = \n')
    # print(str(np.round(N,roundness))+'\n')

    # print('Symmetry of N:')
    # print( np.linalg.norm(N - N.T) )
    # print("\n")

    # print('Skew-symmetry of A\'N:')
    # for Ai in A_list:
    #     print( np.linalg.norm(Ai.T @ N + N.T @ Ai) )
    #     print("\n")