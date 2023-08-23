'''
Tests the spectra of non-symmetric matrices
'''
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import numpy as np
from scipy.linalg import null_space
from common import ret_basis
from examples.integrator_nominalQP import clf_params, cbf_params1, cbf_params2, cbf_params3

n = 2
num_examples = 1000

result_matrix = np.zeros([num_examples, 2])
outliers = {'G': [], 'M': []}
for k in range(num_examples):

    G = np.random.rand(n,n)
    eigG, _ = np.linalg.eig(G)
    G = G.T @ G

    M = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            M = M + np.random.randn()*ret_basis(n, (i,j))
    eigM , _ = np.linalg.eig(M)

    # for b in ret_basis(n):
    #     M = M + np.random.randn()*b
    # eigM , _ = np.linalg.eig(M)

    M_pos_index, = np.where( eigM.real > 0 )
    num_pos_eig_M = len( M_pos_index )
    result_matrix[k, 0] = num_pos_eig_M

    S = G @ M @ G
    eigS , _ = np.linalg.eig(S)

    S_pos_index, = np.where( eigS.real > 0 )
    num_pos_eig_S = len( S_pos_index )
    result_matrix[k, 1] = num_pos_eig_S

    if num_pos_eig_M != num_pos_eig_S:
        outliers['G'].append(G)
        outliers['M'].append(M)

print("Printing the number of positive eigenvalues of: ")
print("M | G M G")
print(result_matrix)

num_outliers = len(outliers['M'])
print("We had "+str(num_outliers)+" outliers. About "+str(100*num_outliers/num_examples)+"% of the data.")
for k in range(num_outliers):
    print("Skew-symmetric operator of M = ")
    M = outliers['M'][k]
    print("M - M.T = \n" + str(M - M.T))
    # print("G = \n" + str(outliers['G'][k]))
    # print("M = \n" + str(outliers['M'][k]))

# G = np.array([[0.15783971, 0.43653843],
#               [0.43653843, 1.2323527 ]])
# M = np.array([[-1.79416954, -2.06094722],
#               [ 2.94104908,  0.2428622 ]])

# eigM , _ = np.linalg.eig(M)
# S = G @ M @ G
# eigS , _ = np.linalg.eig(S)

# print("Eigenvalues of M = " + str(eigM))
# print("Eigenvalues of GMG = " + str(eigS))

print("Conclusion of this experiment: \n")
print("M and GMG do not have, in general, the same number of positive and negative eigenvalues.")