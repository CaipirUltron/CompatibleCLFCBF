'''
Tests the theorems about the eigenspaces of the general Jacobian matrix. 
'''
import numpy as np
from common import symmetric_basis

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

n = 2
m = 3

num_examples = 100
plotting = False

result_matrix = np.zeros([num_examples, 4])
outliers = []
for k in range(num_examples):

    #------- G matrix
    g = np.random.rand(n,m)
    G = g @ g.T

    p1 = np.random.rand(n,n)
    P1 = p1 @ p1.T

    #------- M matrix
    M = np.zeros([n,n])
    for b in symmetric_basis(n):
        M = M + np.random.randn()*b

    alpha = np.random.rand()
    p = np.random.rand()

    z1 = np.random.randn(n)
    z2 = np.random.randn(n)

    hz1 = z1/np.sqrt(z1.T @ G @ z1)
    hz2 = z2/np.sqrt(z2.T @ G @ z2)

    eta = 1 / (1+p*z2.T @ G @ z2)
    beta = 1 - eta

    H = np.diag([eta, beta])
    bH = np.diag([beta, eta])

    Z = np.hstack([hz1, hz2]).reshape(n,2)

    Jac = ( np.eye(n) - G @ Z @ H @ Z.T ) @ G @ M - alpha * G @ Z @ np.diag([1, beta]) @ Z.T

    Q = M @ G @ P1 + P1 @ G @ M

    eigenvals_Jac, eigenvecs_Jac = np.linalg.eig( Jac )
    eigenvals_GM, eigenvecs_GM = np.linalg.eig( G @ M )
    eigenvals_Q, eigenvecs_Q = np.linalg.eig( Q )
    eigenvals_M, eigenvecs_M = np.linalg.eig( M   )

    if plotting and n==3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.quiver( *np.zeros([n,n]), eigenvecs_Q[0,:], eigenvecs_Q[1,:], eigenvecs_Q[2,:], length = 0.1, normalize=True )
        ax.quiver( *np.zeros([n,n]), eigenvecs_M[0,:], eigenvecs_M[1,:], eigenvecs_M[2,:], color = 'r', length = 0.1, normalize=True )

    print(eigenvals_Jac)

    Jac_pos_index, = np.where( eigenvals_Jac.real > 0 )
    num_pos_eig_Jac = len( Jac_pos_index )
    result_matrix[k,0] = num_pos_eig_Jac

    Q_pos_index, = np.where( eigenvals_Q > 0 )
    num_pos_eig_Q = len( Q_pos_index )
    result_matrix[k,1] = num_pos_eig_Q

    GM_pos_index, = np.where( eigenvals_GM > 0 )
    num_pos_eig_GM = len( GM_pos_index )
    result_matrix[k,2] = num_pos_eig_GM

    M_pos_index, = np.where( eigenvals_M > 0 )
    num_pos_eig_M = len( M_pos_index )
    result_matrix[k,3] = num_pos_eig_M

    if (num_pos_eig_Jac == 0) or (num_pos_eig_M == 0):
        outliers.append( result_matrix[k,:] )

print("Printing the number of positive eigenvalues of: ")
print(" Jcl | Q | GM | M ")
print(result_matrix)

print("The following outliers were observed: ")
print(outliers)

plt.show()