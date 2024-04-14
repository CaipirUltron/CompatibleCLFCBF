import math
import json
import itertools

import numpy as np

from scipy.spatial import ConvexHull
from scipy.optimize import fsolve
from shapely.geometry import LineString, LinearRing, Polygon
from shapely.ops import unary_union
from shapely import is_geometry

def cofactor(A):
    """
    Calculate cofactor matrix of A
    """
    sel_rows = np.ones(A.shape[0],dtype=bool)
    sel_columns = np.ones(A.shape[1],dtype=bool)
    CO = np.zeros_like(A)
    sgn_row = 1
    for row in range(A.shape[0]):
        # Unselect current row
        sel_rows[row] = False
        sgn_col = 1
        for col in range(A.shape[1]):
            # Unselect current column
            sel_columns[col] = False
            # Extract submatrix
            MATij = A[sel_rows][:,sel_columns]
            CO[row,col] = sgn_row*sgn_col*np.linalg.det(MATij)
            # Reselect current column
            sel_columns[col] = True
            sgn_col = -sgn_col
        sel_rows[row] = True
        # Reselect current row
        sgn_row = -sgn_row
    return CO

def adjugate(A):
    """
    Calculate adjugate matrix of A
    """
    if A.shape != (2,2):
        raise Exception("Adjugate currently working only in 2D.")
    return np.array([ [ A[1,1], -A[0,1] ], [ -A[1,0], A[0,0] ] ])

def asymmetric_sat(u, limits, slope):
    '''
    Continuous scalar saturation
    '''
    min = limits[0]
    max = limits[1]
    if min > 0:
        raise Exception("Minimum limit should be negative.")
    if max < 0:
        raise Exception("Maximum limit should be positive.")

    t0 = (1/slope)*np.log(-max/min)

    return (max - min)/(1+np.exp( -slope*(u - t0) )) + min

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
        if k == 1:
            k_th_degree_term = np.eye(n, dtype='int64')
        powers_by_degree.append( k_th_degree_term )
        alpha = np.vstack([alpha, k_th_degree_term])

    return alpha, powers_by_degree

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
    matrix_constraints = [ [] for _ in range( p-n-1 ) ]
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
            F[k] = z.T @ E @ z
            matrix_constraints[k-1] = E

    return F, matrix_constraints

def create_quadratic(eigen, R, center, kernel_dim):
    '''
    This function generates the coefficient matrix for a kernel quadratic function
    corresponding to a general quadratic function f(x) = (x-center).T H (x-center)
    Parameters: eigen      -> the list of eigenvalues of H
                R          -> orthonormal matrix with the eigenvectors of H
                center     -> the center point for the quadratic
                kernel_dim -> the dimension of the polynominal kernel function used (standard form - with growing order of monomials)
    Returns: a symmetric, psd, (p,p) matrix representing the quadratic in kernel space (p is the kernel dimension)
    '''
    n = len(eigen)
    if np.shape(R) != (n,n) or len(center) != n:
        raise Exception("Entered dimensions are not correct.")

    eigen = np.array(eigen)    
    center = np.array(center)

    if np.any(eigen < 0.0):
        raise Exception("Quadratic should be positive semi-definite.")

    H = R.T @ np.diag(eigen) @ R
    H = (H + H.T)/2

    std_centered_quadratic = np.zeros([kernel_dim, kernel_dim])
    std_centered_quadratic[0,0] = center.T @ H @ center
    for k in range(n):
        std_centered_quadratic[0,k+1] = -H[k,:].T @ center
        std_centered_quadratic[k+1,0] = -H[k,:].T @ center
    std_centered_quadratic[1:n+1,1:n+1] = H

    return std_centered_quadratic

def circular_boundary_shape( radius, center, kernel_dim ):
    ''' Returns shape matrix of (kernel_dim x kernel_dim) representing a circular obstacle with radius and center '''

    c = 1/(radius**2)
    center = np.array(center)
    n = len(center)

    circular_quadratic = np.zeros([kernel_dim, kernel_dim])
    circular_quadratic[0,0] = c * center.T @ center
    for k in range(n):
        circular_quadratic[0,k+1] = - c * center[k]
        circular_quadratic[k+1,0] = - c * center[k]
    circular_quadratic[1:n+1,1:n+1] = c * np.eye(n)

    return circular_quadratic

def sontag_formula(a, b):
    '''
    General Sontag's formula for stabilization.
    '''
    kappa = 0
    if b != 0:
        kappa = - (a + np.sqrt(a**2 + b**4))/b
    return kappa

def lyap(A: np.ndarray, P: np.ndarray) -> np.ndarray:
    ''' Computes Lyapunov form A P + P A.T '''
    return A @ P + P @ A.T

def compute_curvatures(H, normal):
    '''
    This function computes the maximum/minimum curvatures of a function at a given direction, using a clever algorithm based on the QR decomposition.
    Parameters: H is the Hessian matrix of a given function, 
                normal is the normal vector to the subspace where the optimization is taking place ( normally the direction the function gradient at a given level set )
    Returns: the minimum and maximum curvatures found on the orthogonal subspace
    '''
    n = len(normal)
    if (H.shape[0] != H.shape[1]) or H.shape != (n,n): 
        raise Exception("Dimensions are not consistent.")

    '''
    Generate a random L.I. matrix with the given normal vector at the first column
    '''
    M = np.random.rand(n,n)
    M[:,0] = normal
    while np.abs( np.linalg.det(M) ) <= 1e-10:
        M = np.random.rand(n,n)
        M[:,0] = normal

    '''
    Compute the QR factorization of the previously generated matrix:
    the columns of Q correspond to L.I vectors from Gram-Schmidt orthogonalization
    '''
    Q, R = np.linalg.qr(M)

    barH = Q.T @ H @ Q
    shape_operator = barH[1:,1:]
    curvatures, directions_in_TpS = np.linalg.eig(shape_operator)

    basis_for_TpS = np.zeros([n,n-1])
    for k in range(directions_in_TpS.shape[1]):
        direction = directions_in_TpS[:,k]

        barv = np.zeros(n)
        barv[1:] = direction
        basis_for_TpS[:,k] = Q @ barv

        if basis_for_TpS[:,k].T @ normal > 1e-10:
            raise Exception("Principal curvatures are not computed correctly.")

    return curvatures, basis_for_TpS

def rgb(minimum, maximum, value):
    '''
    Create RGB map from value in interval [ minimum, maximum ].
    Returns: [ r, g, b ] array, with values btw 0-1
    '''
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r

    return [ r/255, g/255, b/255 ]

def innerM(M, v, w):
    '''
    Generalized inner product between vectors v and w. Matrix M must be p.s.d.
    '''
    if not np.all( np.linalg.eigvals(M) >= -1e-10 ):
        raise Exception("Matrix must be positive semi-definite.")
    return v.T @ M @ w

def KKTmultipliers(plant, clf, cbf, x, slack_gain, clf_gain):
    '''
    Returns the KKT multipliers
    '''
    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel

    g = plant.g_method(x)
    G = g @ g.T

    V = clf.function(x)
    m = kernel.function(x)
    Jm = kernel.jacobian(x)

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q

    def inner(v,w): return innerM(Jm @ G @ Jm.T, v, w)

    Pm, Qm, Fm = P @ m, Q @ m, F @ m
    zGz = inner(Pm, Pm) - (inner(Pm, Qm)**2)/(inner(Qm, Qm))
    eta = 1/(1+slack_gain*zGz)

    Const = eta*slack_gain/inner(Qm, Qm)

    l0 = Const * ( inner(Qm, Qm) * ( inner(Pm, Fm) + clf_gain*V ) - inner(Fm, Qm) * inner(Pm, Qm) )
    l1 = Const * ( inner(Pm, Qm) * ( inner(Pm, Fm) + clf_gain*V ) - inner(Fm, Qm) * ( (1/slack_gain) + inner(Pm, Pm) ) )

    return l0, l1

def ellipsoid_parametrization(Q, param):
    '''
    This method implements an angular parametrization of the (rank(Q)-1) dimensional ellipsoid.
    Receives an (rank(Q)-1) dimensional vector of angular parameters for the ellipsoid,
    Returns a corresponding point z in the p-dimensional space at the ellipsoid, that is, z.T @ Q @ z = 1.
    '''
    eigsQ, main_axes = np.linalg.eig(Q)
    axes_lengths = 1/np.sqrt(eigsQ)

    if Q.shape[0] != Q.shape[1]:
        raise Exception("Q must be a square matrix.")
    p = Q.shape[0]
    if np.any(eigsQ < -1e-12):
        raise Exception("Q must be a positive semi-definite matrix.")

    rankQ = np.linalg.matrix_rank(Q, hermitian=True)
    dim_elliptical_manifold = rankQ - 1

    if len(param) != dim_elliptical_manifold:
        raise Exception("Parameter has wrong dimensions")

    reduced_z = np.zeros(rankQ)
    for k in range(rankQ):
        prod = axes_lengths[k]
        if k != dim_elliptical_manifold:
            for i in range(k): 
                prod *= np.sin(param[i])
            reduced_z[k] = prod * np.cos(param[k])
        else:
            for i in range(k): 
                prod *= np.sin(param[i])
            reduced_z[k] = prod

    z = main_axes @ np.array(reduced_z.tolist() + [ 0.0 for _ in range(p-rankQ)])
    return z

def discretize( geom, spacing=0.1 ) -> list[tuple]:
    ''' Returns list of equally spaced points on the lines of the passed Shapely Polygon '''
    
    if is_geometry(geom) and hasattr(geom, "length") and geom.length > 0.0:
    
        # distances = np.arange(0, geom.length, spacing)
        num_pts = round(geom.length/spacing)
        distances = np.linspace(0, geom.length, num_pts)

        if isinstance(geom, LineString) or isinstance(geom, LinearRing):
            points = [geom.interpolate(distance) for distance in distances]
            return [ tuple(pt.coords[0]) for pt in points ]

        if isinstance(geom, Polygon):
            points = [geom.exterior.interpolate(distance) for distance in distances]
            multipoint = unary_union(points)
            return [ tuple(pt.coords[0]) for pt in multipoint.geoms ]
    
    raise Exception("Passed parameter is not a shapely geometry")

def segmentize( segment, pivot ):
    ''' From a single segment of densely spaced points and a center point, split segment into two from the center point '''

    if not isinstance(segment, list): segment.tolist()

    pivot = np.array(pivot)
    distances = [ np.linalg.norm(np.array(pt) - pivot) for pt in segment ]

    min_index = np.argmin(distances)
    closest = np.array(segment[min_index])

    half1 = segment[0:min_index]
    half2 = segment[min_index:]

    dist_to_half1 = sum([ np.linalg.norm(closest - np.array(pt)) for pt in half1 ])
    dist_to_half2 = sum([ np.linalg.norm(closest - np.array(pt)) for pt in half2 ])

    if dist_to_half1 < dist_to_half2:
        segment1 = [ segment[min_index] ] + half1[::-1]
        del half2[0]
        segment2 = [ tuple(pivot) ] + half2

    if dist_to_half1 > dist_to_half2:
        segment1 = [ tuple(pivot) ] + half1[::-1]
        segment2 = half2

    return [segment1, segment2]

def polygon(vertices, spacing=0.1, closed=False) -> list[tuple]:
    ''' Returns list of points forming a polygon with passed vertices and fixed distance between points '''
    
    if closed: line = LinearRing(vertices)
    else: line = LineString(vertices)

    distances = np.arange(0, line.length, spacing)
    points = [line.interpolate(distance) for distance in distances] + ([] if closed else [line.boundary.geoms[1]])
    multipoint = unary_union(points)

    return [ tuple(pt.coords[0]) for pt in multipoint.geoms ]

def box(center, height, width, angle=0, spacing=0.1):
    ''' Returns equally spaced points fitting a box '''
    
    t = (height/2)*np.array([  0, +1 ])
    b = (height/2)*np.array([  0, -1 ])
    l = (width/2 )*np.array([ -1,  0 ])
    r = (width/2 )*np.array([ +1,  0 ])
    tl = t+l
    tr = t+r
    bl = b+l
    br = b+r

    top_left = center + tl @ rot2D( np.deg2rad(angle) )
    top_right = center + tr @ rot2D( np.deg2rad(angle) )
    bottom_left = center + bl @ rot2D( np.deg2rad(angle) )
    bottom_right = center + br @ rot2D( np.deg2rad(angle) )
    
    box_vertices = [ tuple(bottom_left), tuple(bottom_right), tuple(top_right), tuple(top_left) ]
    return polygon( vertices=box_vertices, spacing=spacing, closed=True )

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    # from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    factor = 1                      # must be greater than 1
    centroid = rval.mean(axis=0)
    diffs = rval - centroid
    new_bbox = centroid + factor*diffs

    return new_bbox

def enclosing_circle( rect_limits ):
    ''' Returns radius and center of the circle completely enclosing the passed rectangle '''

    xmin, ymin, xmax, ymax = rect_limits
    xrange = xmax - xmin
    yrange = ymax - ymin
    radius = np.sqrt( xrange**2 + yrange**2 )/2

    vertices = np.array([[xmin, ymin],
                         [xmin, ymax],
                         [xmax, ymin],
                         [xmax, ymax]])
    
    x_c = np.mean(vertices[:, 0])
    y_c = np.mean(vertices[:, 1])
    center = np.array([x_c, y_c])

    return radius, center

def check_kernel(plant, clf, cbf):
    '''
    Basic debugging for equilibrium point algorithms
    '''
    if not hasattr(plant, "kernel"):
        raise Exception("Plant model is not kernel based!")
    kernel = plant.kernel

    if kernel != clf.kernel or kernel != cbf.kernel or clf.kernel != cbf.kernel:
        raise Exception("The plant model, the CLF and the CBF must have the same kernel.")
    
    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")

    if clf._dim != plant.n:
        raise Exception("Plant must have the same dimension as the CLF-CBF pair.")

    return kernel

def clf_function(x, kernel, P):
    '''
    Returns the CLF function value
    '''
    p = kernel.kernel_dim
    if P.shape != (p,p):
        raise Exception("Matrix dimensions are not compatible.")
    z = kernel.function(x)
    return 0.5 * z.T @ P @ z

def vecQ(x, kernel, Q):
    '''
    vecQ function for the computation of the invariant set / equilibrium points.
    '''
    p = kernel.kernel_dim
    if Q.shape != (p,p):
        raise Exception("Matrix dimensions are not compatible.")
    
    z = kernel.function(x)
    A_list = kernel.get_A_matrices()
    n = len(A_list)

    vecQ = np.empty((n,), dtype=float)
    for k in range(n): vecQ[k] = z.T @ A_list[k].T @ Q @ z

    # vecQ_list = [ z.T @ A_list[k].T @ Q @ z for k in range(len(A_list)) ]
    return vecQ

def vecP(x, kernel, P, F, params):
    '''
    vecP function for the computation of the invariant set / equilibrium points.
    '''
    p = kernel.kernel_dim
    if P.shape != (p,p) or F.shape != (p,p):
        raise Exception("Matrix dimensions are not compatible.")
    
    z = kernel.function(x)
    A_list = kernel.get_A_matrices()
    n = len(A_list)

    V = clf_function(x, kernel, P)

    vecP = np.empty((n,), dtype=float)
    for k in range(n): vecP[k] = z.T @ A_list[k].T @ ( params["slack_gain"] * params["clf_gain"] * V * P - F ) @ z

    # vecP_list = [ z.T @ A_list[k].T @ ( params["slack_gain"] * params["clf_gain"] * V * P - F ) @ z for k in range(len(A_list)) ]
    return vecP

def det_invariant(x, kernel, P, Q, F, params):
    '''
    Returns the determinant det([ vecQ, vecP ]) for a given point x and CLF-CBF pair.
    '''
    n = len(x)
    # vecQ_a = np.array( vecQ(x, kernel, Q) )
    # vecP_a = np.array( vecP(x, kernel, P, F, params) )
    # W = np.hstack([vecQ_a.reshape(n,1), vecP_a.reshape(n,1)])

    W = np.hstack([ vecQ(x, kernel, Q).reshape(n,1), vecP(x, kernel, P, F, params).reshape(n,1) ])

    return np.linalg.det(W)
    # return np.sqrt( np.linalg.det( W.T @ W ) ) # does not work

def lambda_invariant(x, kernel, P, Q, F, params):
    '''
    Returns the lambda corresponding to a point on the invariant set det([ vecQ, vecP ]) = 0.
    '''
    vecQ_a = vecQ(x, kernel, Q)
    vecP_a = vecP(x, kernel, P, F, params)
    return (vecQ_a.T @ vecP_a) / np.linalg.norm(vecQ_a)**2

def L(x, kernel, P, Q, F, params):
    '''
    Returns L matrix: L = F + l Q - p gamma V(x,P) P
    '''
    l = lambda_invariant(x, kernel, P, Q, F, params)
    return F + l * Q - params["slack_gain"] * params["clf_gain"] * clf_function(x, kernel, P) * P

def S(x, kernel, P, Q, plant, params):
    '''
    Returns the S matrix: S = H(x,l,P) - (1/pgV^2) * fc fc.T, for stability computation of equilibrium points 
    '''
    A_list = kernel.get_A_matrices()
    n = len(A_list)
    if len(x) != n:
        raise Exception("Dimensions are incorrect.")

    V = clf_function(x, kernel, P)
    z = kernel.function(x)

    fc = plant.get_fc(x)
    F = plant.get_F()

    L_matrix = L(x, kernel, P, Q, F, params)
    S_matrix = [ [ z.T @ A_list[i].T @ ( L_matrix @ A_list[j] + A_list[j].T @ L_matrix ) @ z - fc[i]*fc[j]/(params["slack_gain"] * params["clf_gain"] * (V**2)) for j in range(n) ] for i in range(n) ]
    return np.array(S_matrix)

def add_to(point, l, *connections):
    '''
    Adds point to list l, if new. point is the default dict for points in the invariant set 
    '''
    if len(connections) > 1: raise Exception("Add accepts only 1 optional argument.")
    pt = np.array(point["x"])

    if len(l) > 0:
        costs = np.linalg.norm( pt - np.array([ elem["x"] for elem in l ]), axis=1 )
        indexes = (costs < 1e-3).nonzero()[0]
        if len(indexes) > 0:   # pt is old
            pt_list_index = indexes[0]
        else:                      # pt is new
            l.append(point)
            pt_list_index = len(l)-1
    else: 
        l.append(point)
        pt_list_index = len(l)-1
    if len(connections) == 1: connections[0].append(pt_list_index)

def show_message(pts, text):
    '''
    Show message for points in the invariant set. 
    '''
    num_pts = len(pts)
    if num_pts > 0:
        print(f"Found {num_pts} {text} at:")
    else:
        print(f"Found {num_pts} {text}.")
    for sol in pts:
        if "x" in sol.keys():
            x = sol["x"]
            l = sol["lambda"]
            h = sol["h"]
            gradh = sol["nablah"]
            output_text = "x = " + str(x) + ", lambda = " + str(l) + ", h = " + str(h) + ", ||∇h|| = " + str(gradh)
            if "equilibrium" in sol.keys() and "stability" in sol.keys():
                type_of = sol["equilibrium"]
                stability = sol["stability"]
                output_text += ", equilibrium is " + str(type_of) + " (" + str(stability) + ")"
            if "rem_by_minimizer" in sol.keys():
                output_text += ", rem_by_minimizer " + str(sol["rem_by_minimizer"])
            if "rem_by_maximizer" in sol.keys():
                output_text += ", rem_by_maximizer " + str(sol["rem_by_maximizer"])
            if "type" in sol.keys():
                output_text += ", type = " + str(sol["type"])
        print(output_text)

def load_compatible(file_name, P, load_compatible=True):
    '''
    Loads the shapes represented by matrices P if file exists.
    flag = True loads the compatible shape.
    flag = False loads the original (incompatible) shape.
    '''
    try:
        with open("logs/" + file_name.split('/')[-1].replace(".py","") + "_comp.json") as file:

            message = "Loading compatibilization file. "
            compatible_dict = json.load(file)

            message += "Original P is "
            if compatible_dict["is_original_compatible"]: message += "compatible. "
            else: message += "incompatible. "

            message += "Processed P is "
            if compatible_dict["is_processed_compatible"]: message += "compatible. "
            else: message += "incompatible. "

            print(message)

            if load_compatible: 
                print("Loading processed P matrix.")
                return np.array(compatible_dict["P_processed"])
            print("Loading original P matrix.")
            return np.array(compatible_dict["P_original"])
        
    except IOError:
        print("Couldn't locate compatibilization file. Returning passed P matrix.")
        return P

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