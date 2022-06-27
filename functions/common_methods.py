import numpy as np
from scipy.optimize import fsolve

def vector2triangular(vector):
    '''
    Transforms numpy vector to corresponding upper triangular matrix.
    '''
    dim = len(vector)
    if dim < 3:
        raise Exception("The input vector must be of length 3 or higher.")
    n = int((-1 + np.sqrt(1+8*dim))/2)
    tri_basis = triangular_basis(n)
    T = np.zeros([n,n])
    for k in range(dim):
        T = T + tri_basis[k]*vector[k]
    return T

def vector2sym(vector):
    '''
    Transforms numpy vector to corresponding symmetric matrix.
    '''
    dim = len(vector)
    if dim < 3:
        raise Exception("The input vector must be of length 3 or higher.")
    n = int((-1 + np.sqrt(1+8*dim))/2)
    sym_basis = symmetric_basis(n)
    S = np.zeros([n,n])
    for k in range(dim):
        S = S + sym_basis[k]*vector[k]
    return S

def triangular2vector(T):
    '''
    Stacks the coefficients of an upper triangular matrix to a numpy vector.
    '''
    n = T.shape[0]
    if n < 2:
        raise Exception("The input matrix must be of size 2x2 or higher.")
    tri_basis = triangular_basis(n)
    dim = int((n*(n+1))/2)
    vector = np.zeros(dim)
    for k in range(dim):
        list = np.nonzero(tri_basis[k])
        i, j = list[0][0], list[1][0]
        vector[k] = T[i][j]
    return vector

def sym2vector(S):
    '''
    Stacks the coefficients of a symmetric matrix to a numpy vector.
    '''
    n = S.shape[0]
    if n < 2:
        raise Exception("The input matrix must be of size 2x2 or higher.")
    sym_basis = symmetric_basis(n)
    dim = int((n*(n+1))/2)
    vector = np.zeros(dim)
    for k in range(dim):
        list = np.nonzero(sym_basis[k])
        i, j = list[0][0], list[1][0]
        vector[k] = S[i][j]
    return vector

def triangular_basis(n):
    '''
    Returns the canonical basis of the space of upper triangular (n x n) matrices.
    '''
    tri_basis = list()
    EYE = np.eye(n)
    for i in range(n):
        for j in range(i,n):
            if i == j:
                tri_basis.append(np.outer(EYE[:,i], EYE[:,j]))
            else:
                tri_basis.append(np.outer(EYE[:,i], EYE[:,j]))
    return tri_basis

def symmetric_basis(n):
    '''
    Returns the canonical basis of the space of symmetric (n x n) matrices.
    '''
    sym_basis = list()
    EYE = np.eye(n)
    for i in range(n):
        for j in range(i,n):
            if i == j:
                sym_basis.append(np.outer(EYE[:,i], EYE[:,j]))
            else:
                sym_basis.append(np.outer(EYE[:,i], EYE[:,j]) + np.outer(EYE[:,j], EYE[:,i]))
    return sym_basis

def skewsymmetric_basis(n):
    '''
    Returns the canonical basis of the space of skew-symmetric (n x n) matrices.
    '''
    sym_basis = list()
    EYE = np.eye(n)
    for i in range(n):
        for j in range(i,n):
            if i != j:
                sym_basis.append( ((-1)**(i+j))*(np.outer(EYE[:,i], EYE[:,j])-np.outer(EYE[:,j], EYE[:,i])) )
    return sym_basis

def hat(omega):
    '''
    Returns the skew-symmetric matrix corresponding to the omega vector array.
    '''
    n = len(omega)
    basis = skewsymmetric_basis(n)
    omega_hat = np.zeros([n,n])
    for k in range(len(basis)):
        omega_hat = omega_hat + omega[k]*basis[k]

def rot2D(theta):
    '''
    Standard 2D rotation matrix.
    '''
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s),(s,c)))
    return R

def canonical2D(eigen, theta):
    '''
    Returns the (2x2) symmetric matrix with eigenvalues eigen and eigenvector angle theta.
    '''
    Diag = np.diag(eigen)
    R = rot2D(theta)
    H = R @ Diag @ R.T
    return H

def sym2triangular(P):
    '''
    This function decomposes a symmetric semidefinite matrix P into the form P = L'L.
    '''
    eigs, Q = np.linalg.eig(P)
    if np.any(eigs*eigs[0]) < 0:
        Exception("The input matrix is not definite.")
    Linit = np.diag(np.sqrt(eigs)) @ Q.T
    param_init = triangular2vector(Linit)
    dimension = len(eigs)

    def func(l):
        f = []
        L = vector2triangular(l)
        for i in range(dimension):
            for j in range(dimension):
                if i <= j:
                    l_term = 0
                    for k in range(dimension):
                        l_term = l_term + L[k,i] * L[k,j]
                    f.append( l_term - P[i,j] )
        return f

    param_sol = fsolve(func, param_init)
    return vector2triangular( param_sol )