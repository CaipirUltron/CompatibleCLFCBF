'''
Tests the theorems about the eigenspaces of the general Jacobian matrix. 
'''
import sys
import json
import time
import numpy as np
from common import symmetric_basis
from scipy.linalg import null_space

n = 3
m = 3

num_examples = 100

result_matrix = np.zeros([num_examples, 2])
outliers = {'Jac': [], 'G M': []}
for k in range(num_examples):

    #------- G matrix
    g = np.random.rand(n,m)
    G = g @ g.T

    #------- M matrix
    M = np.random.rand(n,n)
    M = np.zeros([n,n])
    for b in symmetric_basis(n):
        M = M + np.random.randn()*b

    alpha = np.random.rand()
    p = np.random.rand()

    z1 = np.random.rand(n)
    z2 = np.random.rand(n)

    hz1 = z1/np.sqrt(z1.T @ G @ z1)
    hz2 = z2/np.sqrt(z2.T @ G @ z2)

    eta = 1 / (1+p*z2.T @ G @ z2)
    beta = 1 - eta

    H = np.diag([eta, beta])
    bH = np.diag([beta, eta])

    Z = np.hstack([hz1, hz2]).reshape(n,2)

    Jac = ( np.eye(n) - G @ Z @ H @ Z.T ) @ G @ M - alpha * G @ Z @ np.diag([1, beta]) @ Z.T

    eigenvals_Jac, eigenvecs_Jac = np.linalg.eig( Jac )
    eigenvals_GM, eigenvecs_GM = np.linalg.eig( G @ M )

    Jac_pos_index, = np.where( eigenvals_Jac.real > 0 )
    num_pos_eig_Jac = len( Jac_pos_index )
    result_matrix[k,0] = num_pos_eig_Jac

    GM_pos_index, = np.where( eigenvals_GM.real > 0 )
    num_pos_eig_GM = len( GM_pos_index )
    result_matrix[k,1] = num_pos_eig_GM

print("Printing the number of positive eigenvalues of: ")
print("Jac | G M")
print(result_matrix)