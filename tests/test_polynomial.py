'''
Test polynomial functions
'''
import numpy as np
from functions import PolynomialFunction
from common import *

n = 2
max_degree = 2

# P = np.random.rand(p,p)
# P = P.T @ P
# eigP , _= np.linalg.eig(P)
# P = P/np.max(eigP)

state = np.random.rand(n)
clf = PolynomialFunction(*state, degree = max_degree)

print(clf.get_kernel())
# print( generate_monomial_list(n, max_degree) )

A_list = clf.get_A_matrices()

# N1 = np.array([ [ 1,0,0,0,0,0 ], 
#                 [ 0,0,0,0,0,0 ],
#                 [ 0,0,0,0,0,0 ], 
#                 [ 0,0,0,0,0,0 ], 
#                 [ 0,0,0,0,0,0 ], 
#                 [ 0,0,0,0,0,0 ] ])

# N2 = np.array([ [ 0,0,1,1,0,0 ], 
#                 [ 0,0,0,0,1,0 ],
#                 [ 1,0,0,0,0,0 ], 
#                 [ 1,0,0,0,0,0 ], 
#                 [ 0,1,0,0,0,1 ], 
#                 [ 0,0,0,0,1,0 ] ])

N_list = clf.get_N_matrices()
roundness = 5
for k in range(len(N_list)):
    N = N_list[k]/np.max(N_list[k])

    print('Symmetry:')
    print( np.linalg.norm(N - N.T) )

    print('Skew-symmetric:')
    for Ai in A_list:
        print( np.round(Ai.T @ N,roundness) )
        print("\n")

    print('N'+str(k+1)+' = \n')
    print(str(np.round(N,roundness))+'\n')