import math
import numpy as np
import itertools
from scipy.optimize import fsolve

def sat(u, limits):
    '''
    Scalar saturation.
    '''
    min = limits[0]
    max = limits[1]
    if u > max:
        return max
    if u < min:
        return min
    return u

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

def ret_basis(n, index_tuple):
    '''
    Returns the a n x n matrix of zeros, with 1 at index_tuple, where index_tuple indexation starts from 1 
    '''
    EYE = np.eye(n)
    i, j = index_tuple[0], index_tuple[1]
    if i < 1 or j < 1:
        raise Exception("Indexation starts from 1") 
    if i > n:
        raise Exception("Index out of bounds for axis 0") 
    if j > n:
        raise Exception("Index out of bounds for axis 1")
    
    return np.outer(EYE[:,i-1], EYE[:,j-1])

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

def rot3D(theta, axis):
    '''
    Angle and axis 3D rotation matrix.
    '''
    axis_norm = np.linalg.norm(axis)
    u = axis/axis_norm
    uut = np.outer(u, u)
    u_cross = np.cross(u,np.identity(u.shape[0])*-1)
    cos, sin = np.cos(theta), np.sin(theta)
    R = cos*np.eye(3) + sin*u_cross + (1-cos)*uut
    return R

def canonical2D(eigen, theta):
    '''
    Returns the (2x2) symmetric matrix with eigenvalues eigen and eigenvector angle theta.
    '''
    Diag = np.diag(eigen)
    R = rot2D(theta)
    H = R @ Diag @ R.T
    return H

def canonical3D(eigen, theta, axis):
    '''
    Returns the (3x3) symmetric matrix with eigenvalues eigen and eigenvector angle theta.
    '''
    Diag = np.diag(eigen)
    R = rot3D(theta, axis)
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

def num_comb(n, d):
    '''
    Returns the number of monomials of n-dimensions up to degree d
    '''
    return math.comb(n+d,d)

def generate_monomial_list(n, max_degree):
    '''
    Returns the matrix of monomial powers of dimension n up to degree max_degree, with terms in increasing order of degree.
    '''
    powers = np.array(list(itertools.product(*[range(max_degree+1)]*n))) # all possible powers of each variable up to degree max_degree

    # Stores terms by order of maximum powers, up to max_degree
    alpha = np.array([], dtype='int64').reshape(0,n)
    powers_by_degree = []
    for k in range(max_degree+1):
        k_th_degree_term = powers[np.where(np.fromiter(map(sum, powers),dtype='int64')==k)]
        powers_by_degree.append( k_th_degree_term )
        alpha = np.vstack([alpha, k_th_degree_term])

    return alpha, powers_by_degree

def kernel_constraints( z, terms_by_degree ):
    '''
    This algorithm generates the matrices needed to implement all constraints keeping a vector z 
    inside the kernel space of a polynomial kernel represented by powers_by_degree.
    '''
    max_degree = len(terms_by_degree)-1
    n, p = np.shape(terms_by_degree[0])[1], 0
    for k_th_degree_term in terms_by_degree:
        k_th_shape = np.shape(k_th_degree_term)
        if n != k_th_shape[1]:
            raise Exception("List of terms is invalid.")
        p += k_th_shape[0]

    # Function F(z) is the collection of p-n+1 kernel constraints
    F = np.zeros(p-n+1)
    F[0] = np.eye(p)[0,:] @ z - 1.0

    k = 0
    # matrix_constraints = []
    for degree in range(2,max_degree+1):
        for term in terms_by_degree[degree]:
            k += 1

            # If degree = 2, just look at the terms of degree - 1 (first degree terms)
            if degree == 2:
                first_degree_terms = terms_by_degree[1]
                break_flag = False
                for k1 in range(n):
                    for k2 in range(n):
                        if np.all(term == first_degree_terms[k1,:] + first_degree_terms[k2,:]):
                            i, j = k1+2, k2+2
                            break_flag = True
                            break
                    if break_flag: break

            # If degree > 2, look at the terms of degree-1 and degree-2, and build combinations of them 
            if degree > 2:
                d_minus_1_terms = terms_by_degree[degree-1]
                d_minus_2_terms = terms_by_degree[degree-2]
                break_flag = False
                for k1 in range(len(d_minus_1_terms)): # first loop: terms of degree - 1
                    for k2 in range(len(d_minus_2_terms)):  # second loop: terms of degree - 2
                        if np.all( term == d_minus_1_terms[k1,:] + d_minus_2_terms[k2,:] ):
                            i = k1+num_comb(n, degree-2)+1
                            j = k2+num_comb(n, degree-3)+1
                            break_flag = True
                            break
                    if break_flag: break

            E = ret_basis(p, (1,k+n+1)) - ret_basis(p, (i,j))
            # matrix_constraints.append( E )
            F[k] = z.T @ E @ z

    return F

def generate_monomials_from_symbols(symbol_list, alpha):
    '''
    Returns the vector of monomial powers corresponding to alpha, with symbols given by symbol_list
    '''
    n = len(symbol_list)
    monomials = []
    for row in alpha:
        mon = 1
        for dim in range(n):
            mon = mon*symbol_list[dim]**row[dim]
        monomials.append( mon )
    return monomials

class Rect():
    '''
    Simple rectangle.
    '''
    def __init__(self, sides, center_offset):
        self.length = sides[0]
        self.width = sides[1]
        self.center_offset = center_offset

    def get_center(self, pose):
        x, y, angle = pose[0], pose[1], pose[2]
        return ( x - self.center_offset*np.cos(angle), y - self.center_offset*np.sin(angle) )

    def get_corners(self, pose, *args):
        x, y, angle = pose[0], pose[1], pose[2]

        topleft = ( rot2D(angle) @ np.array([-self.length/2, self.width/2]) + np.array( self.get_center(pose) ) ).tolist()
        topright = ( rot2D(angle) @ np.array([ self.length/2, self.width/2]) + np.array( self.get_center(pose) ) ).tolist()
        bottomleft = ( rot2D(angle) @ np.array([-self.length/2, -self.width/2]) + np.array( self.get_center(pose) ) ).tolist()
        bottomright = ( rot2D(angle) @ np.array([ self.length/2, -self.width/2]) + np.array( self.get_center(pose) ) ).tolist()

        if len(args):
            if 'topleft' in args[0].lower():
                return topleft
            elif 'topright' in args[0].lower():  
                return topright
            elif 'bottomleft' in args[0].lower():
                return bottomleft
            elif 'bottomright' in args[0].lower():
                return bottomright
            else:
                return [ topleft, topright, bottomleft, bottomright ]
        else:
            return [ topleft, topright, bottomleft, bottomright ]
        
# def generate_monomial_list(n, d):
#     '''
#     Returns the matrix of monomial powers of dimension n up to degree d.
#     '''
#     to_be_removed = []
#     combinations = list( itertools.product( list(range(d+1)), repeat=n ) )
#     for k in range(len(combinations)):
#         if sum(combinations[k])>d:
#             to_be_removed.append(k)
#     for ele in sorted(to_be_removed, reverse = True):
#         del combinations[ele]

#     alpha = np.array(combinations)
#     p = num_comb(n, d)
#     EYE = np.eye(n, dtype=int)
#     for i in range(n):
#         row = EYE[i,:]
#         for j in range(1,p):
#             if np.all(alpha[j,:] == row):
#                 aux = alpha[i+1,:]
#                 alpha[j,:] = aux
#                 alpha[i+1,:] = row
#                 break

#     return alpha