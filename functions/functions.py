import time
import itertools
import numpy as np
import scipy as sp
import sympy as sym
import cvxpy as cp
import logging
import warnings

import operator
import contourpy as ctp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from math import comb
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.optimize import minimize
from shapely import geometry, intersection

from common import *
from dynamic_systems import Integrator, KernelAffineSystem

def are_all_type( element_list, set_of_types ):
    ''' 
    Tests if all elements of element_list are of the types on set_of_types.
    Raises an error if it detects two or more elements of different type.
    '''
    elem_types_insertance = [ isinstance(elem, set_of_types) for elem in element_list ]

    all_are_of_set_of_types = np.all(elem_types_insertance)
    if (not all_are_of_set_of_types) and np.any(elem_types_insertance):
        raise Exception("Coefficients are not all of the same type.")
    return all_are_of_set_of_types

def commutation_matrix(n):
    '''
    Generate commutation matrix K relating the vectorization of a matrix n x n matrix A with the vectorization of its transpose A', as
    vec(A') = K vec(A).
    '''
    # determine permutation applied by K
    w = np.arange(n * n).reshape((n, n), order="F").T.ravel(order="F")
    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(n * n)[w, :]

def vec(A):
    '''
    Vectorize matrix in a column-major form (Fortran-style).
    '''
    return A.flatten('F')

def mat(vec):
    '''
    De-vectorize a square matrix which was previously vectorized in a column-major form.
    '''
    n = np.sqrt(len(vec))
    if (not n.is_integer()):
        raise Exception('Input vector does not represent a vectorized square matrix.')
    n = int(n)
    return vec.reshape(n,n).T

class Function(ABC):
    ''' 
    Implementation of abstract class for scalar functions of any input dimension
    '''
    def __init__(self, **kwargs):

        # Initialize basic parameters
        self._dim = 2
        self._output_dim = 1
        self.color = mcolors.BASE_COLORS["k"]
        self.linestyle = "solid"
        self.limits = (-1,1,-1,1)
        self.spacing = 0.1

        self.set_params(**kwargs)

        if self._output_dim == 1:
            self.generate_contour()

    def _validate(self, point):
        ''' Validates input data '''
        if not isinstance(point, (list, np.ndarray)): raise Exception("Input data point is not a numeric array.")
        if isinstance(point, list): point = np.array(point)
        return point

    @abstractmethod
    def _function(self, point: np.ndarray) -> np.ndarray:
        '''
        Abstract implementation of function computation. 
        Must receive point as input and return the corresponding function value.
        Overwrite on children classes.
        '''
        pass

    @abstractmethod
    def _gradient(self, point: np.ndarray) -> np.ndarray:
        '''
        Abstract implementation of gradient computation. 
        Must receive point as input and return the corresponding gradient value.
        Overwrite on children classes.
        '''
        pass

    @abstractmethod
    def _hessian(self, point: np.ndarray) -> np.ndarray:
        '''
        Abstract implementation of hessian computation. Must receive point as input and return the corresponding hessian value.
        Overwrite on children classes.
        '''
        pass

    def generate_contour(self):
        '''
        Create contour generator object for the given function.
        Parameters: limits (2x2 array) - min/max limits for x,y coords
                    spacing - grid spacing for contour generation
        '''        
        if self._dim != 2:
            logging.warning("Contour plot can only be used for 2D functions.")
            self.contour = None
            return

        x_min, x_max, y_min, y_max = self.limits
        x = np.arange(x_min, x_max, self.spacing)
        y = np.arange(y_min, y_max, self.spacing)
        xg, yg = np.meshgrid(x,y)
        
        fvalues = np.zeros(xg.shape)
        for i,j in itertools.product(range(xg.shape[0]), range(xg.shape[1])):
            pt = np.array([xg[i,j], yg[i,j]])
            fvalues[i,j] = self.function(pt)
        
        self.contour = ctp.contour_generator(x=xg, y=yg, z=fvalues )

    def function(self, point):
        return self._function(self._validate(point))

    def gradient(self, point):
        return self._gradient(self._validate(point))    

    def hessian(self, point):
        return self._hessian(self._validate(point))

    def set_params(self, **params):
        ''' Sets function basic parameters (mostly plotting) '''

        for key in params.keys():
            key = key.lower()
            if key == "dim":
                self._dim = params["dim"]
                continue
            if key == "color":
                self.color = params["color"]
                continue
            if key == "linestyle":
                self.linestyle = params["linestyle"]
                continue
            if key == "limits":
                self.limits = params["limits"]
                continue
            if key == "spacing":
                self.spacing = params["spacing"]
                continue

    def get_levels(self, levels=[0.0] ) -> list:
        ''' Generates function level sets from the contour generator object '''
        if not self.contour: return []

        level_contours = []
        for lvl in levels:
            line = self.contour.lines(lvl)
            level_contours.append(line)
        return level_contours

    def plot_levels(self, ax = plt, levels=[0.0], **kwargs):
        ''' Plots function level sets at the input axis ax. Additional args may be passed for color and linestyle '''
        
        color = self.color
        linestyle = self.linestyle
        for key in kwargs.keys():
            key = key.lower()
            if key == "color":
                color = kwargs["color"]
                continue
            if key == "linestyle":
                linestyle = kwargs["linestyle"]
                continue

        collections = []
        for level in self.get_levels(levels):
            for segment in level:
                line2D = ax.plot( segment[:,0], segment[:,1], color=color, linestyle=linestyle )
                collections.append(line2D[0])
        return collections

class Quadratic(Function):
    '''
    Class for quadratic function representing x'Ax + b'x + c = 0.5 (x - p)'H(x-p) + height = 0.5 x'Hx - 0.5 p'(H + H')x + 0.5 p'Hp + height
    '''
    def __init__(self, *args):

        # Set parameters
        super().__init__(*args)

        if self._dim > 1:
            self.A = np.zeros([self._dim,self._dim])
            self.b = np.zeros(self._dim)
            self.critical_point = np.zeros(self._dim)
            self.dcritical = np.zeros(self._dim)
        else:
            self.A = 0.0
            self.b = 0.0
            self.critical_point = 0.0
            self.dcritical = 0.0
        self.c = 0.0
        self.height = 0.0

        # self.set_param(kwargs)

        # Set eigenbasis for hessian matrix
        _, _, Q = self.compute_eig()
        self.eigen_basis = np.zeros([self._dim, self._dim, self._dim])
        for k in range(self._dim):
            self.eigen_basis[:][:][k] = np.outer( Q[:,k], Q[:,k] )

    def set_param(self, **kwargs):
        '''
        Sets the quadratic function parameters.
        '''
        for key in kwargs:
            if key == "hessian":
                self._hessian = np.array(kwargs[key])
            if key == "critical":
                self.critical_point = np.array(kwargs[key])
            if key == "height":
                self.height = kwargs[key]
            if key == "dcritical":
                self.dcritical = np.array(kwargs[key])

        self.A = 0.5 * self._hessian
        self.b = - 0.5*( self._hessian + self._hessian.T ) @ self.critical_point
        self.c = 0.5 * self.critical_point @ ( self._hessian @ self.critical_point ) + self.height

        for key in kwargs:
            if key == "A":
                self.A = kwargs[key]
            if key == "b":
                self.b = kwargs[key]
            if key == "c":
                self.c = kwargs[key]

    def function(self, point):
        '''
        General quadratic function.
        '''
        return np.array(point) @ ( self.A @ np.array(point) ) + self.b @ np.array(point) + self.c

    def gradient(self, point):
        '''
        Gradient of general quadratic function.
        '''
        return ( self.A + self.A.T ) @ np.array(point) + self.b

    def hessian(self, point):
        '''
        Hessian of general quadratic function.
        '''
        return ( self.A + self.A.T )

    def eigen2hessian(self, eigen):
        '''
        Returns hessian matrix from a given set of eigenvalues.
        '''
        if self._dim != len(eigen):
            raise Exception("Dimension mismatch.")

        H = np.zeros(self._dim)
        for k in range(self._dim):
            H = H + eigen[k] * self.eigen_basis[:][:][k]

        return H

    def get_critical(self):
        return self.critical_point

    def get_height(self):
        return self.height

    def get_critical_derivative(self):
        return self.dcritical

    def compute_eig(self):
        eigen, Q = np.linalg.eig(self.hessian(0))
        angle = np.arctan2(Q[0, 1], Q[0, 0])
        return eigen, angle, Q

class QuadraticLyapunov(Quadratic):
    '''
    Class for Quadratic Lyapunov functions of the type (x-x0)'Hv(x-x0), parametrized by vector pi_v.
    Here, the Lyapunov minimum is a constant vector x0, and the hessian Hv is positive definite and parametrized by:
    Hv = Lv(pi_v)'Lv(pi_v) + epsilon I_n (Lv is upper triangular and epsilon is a small positive constant).
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        super().set_param(**kwargs)
        super().set_param(height=0.0)

        self.epsilon = 0.0
        self.Lv = sym2triangular( self._hessian-self.epsilon*np.eye(self._dim) )
        self.param = triangular2vector( self.Lv )
        self.dynamics = Integrator(self.param,np.zeros(len(self.param)))

    def get_param(self):
        '''
        This function gets the params corresponding to the Lyapunov Hessian matrix.
        '''
        return self.param

    def set_epsilon(self, epsilon):
        '''
        Sets the minimum eigenvalue for the Lyapunov Hessian matrix.
        '''
        self.epsilon = epsilon
        self.set_param(self.param)

    def set_param(self, param):
        '''
        Sets the Lyapunov function parameters.
        '''
        self.param = param
        Lv = vector2triangular(param)
        Hv = Lv.T @ Lv + self.epsilon*np.eye(self._dim)
        super().set_param(hessian = Hv)

    def set_critical(self, pt):
        '''
        Sets the Lyapunov function critical point.
        '''
        super().set_param(critical = pt)

    def set_critical_derivative(self, dcritical):
        '''
        Sets the derivative of the Lyapunov function critical point.
        '''
        super().set_param(dcritical = dcritical)

    def update(self, param_ctrl, dt):
        '''
        Integrates the parameters.
        '''
        self.dynamics.set_control(param_ctrl)
        self.dynamics.actuate(dt)
        self.set_param(self.dynamics.get_state())

    def get_partial_Hv(self):
        '''
        Returns the partial derivatives of Hv wrt to the parameters.
        '''
        tri_basis = triangular_basis(self._dim)
        partial_Hv = np.zeros([ len(self.param), self._dim, self._dim ])
        for i in range(len(self.param)):
            for j in range(len(self.param)):
                partial_Hv[i,:,:] = partial_Hv[i,:,:] + ( tri_basis[i].T @ tri_basis[j] + tri_basis[j].T @ tri_basis[i] )*self.param[j]

        return partial_Hv

class QuadraticBarrier(Quadratic):
    '''
    Class for Quadratic barrier functions. For positive definite Hessians, the unsafe set is described by the interior of an ellipsoid.
    The symmetric Hessian is parametrized by Hh(pi) = sum^n_i Li pi_i, where {Li} is the canonical basis of the space of (n,n) symmetric matrices.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        super().set_param(**kwargs)
        super().set_param(height = -0.5)

        self.param = sym2vector(self._hessian)
        self.dynamics = Integrator(self.param,np.zeros(len(self.param)))
        self.last_gamma_sol = 0.0
        self.gamma_min = -np.pi/2
        self.gamma_max = +np.pi/2

    def get_param(self):
        '''
        This function gets the params corresponding to the barrier Hessian matrix.
        '''
        return self.param

    def set_param(self, param):
        '''
        Sets the barrier function parameters.
        '''
        self.param = param
        super().set_param(hessian = vector2sym(param))

    def set_critical(self, pt):
        '''
        Sets the barrier function critical point.
        '''
        super().set_param(critical = pt)

    def update(self, param_ctrl, dt):
        '''
        Integrates the barrier function parameters.
        '''
        self.dynamics.set_control(param_ctrl)
        self.dynamics.actuate(dt)
        self.set_param(self.dynamics.get_state())

    def barrier_set(self, set_parameters):
        '''
        Computes the barrier with respect to a set define by set_parameters
        '''
        r = set_parameters["radius"]
        p_center = set_parameters["center"]
        theta = set_parameters["orientation"]

        def compute_pt(gamma):
            return np.array(p_center) + r*rot2D(theta) @ np.array([np.cos(gamma), np.sin(gamma)]).reshape(2)

        def necessary_cond(gamma):
            p_gamma = compute_pt( gamma )
            nablah = self.evaluate_gradient(*p_gamma)[0]
            return nablah.dot(np.array([-np.sin(theta+gamma), np.cos(theta+gamma)]))

        def cost(gamma):
            p_gamma = compute_pt(gamma)
            return self.evaluate_function(*p_gamma)[0]

        # Solve optimization problem
        import scipy.optimize as opt
        results = opt.minimize( cost, self.last_gamma_sol )
        gamma_opt = results.x[0]

        h = cost(gamma_opt)

        p_gamma_opt = compute_pt( gamma_opt )
        nablah = self.evaluate_gradient(*p_gamma_opt)[0]

        self.last_gamma_sol = gamma_opt

        return h, nablah, p_gamma_opt, gamma_opt

class Gaussian(Function):
    '''
    Class for gaussian function of the type N(x) = c exp( -0.5 (x-mu).T Sigma (x-mu) )
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self.c = 0.0
        if self._dim > 1:
            self.mu = np.zeros(self._dim)
            self.Sigma = np.zeros([self._dim,self._dim])
        else:
            self.mu = 0.0
            self.Sigma = 0.0
        self.set_param(**kwargs)

        self.epsilon = 0.0
        self.Lv = sym2triangular( self._hessian-self.epsilon*np.eye(self._dim) )
        self.param = triangular2vector( self.Lv )
        self.dynamics = Integrator(self.param,np.zeros(len(self.param)))

    def set_param(self, **kwargs):
        '''
        Sets the quadratic function parameters.
        '''
        for key in kwargs:
            if key == "constant":
                self.c = kwargs[key]
            if key == "mean":
                self.mu = np.array(kwargs[key])
            if key == "shape":
                if np.shape(kwargs[key]) != ( self._dim, self._dim ):
                    raise Exception('Shape matrix must be the same dimension as the mean.')
                self.Sigma = np.array(kwargs[key])

    def update(self, param_ctrl, dt):
        pass

    def function(self, point):
        '''
        Gaussian function.
        '''
        v = np.array(point) - self.mu
        return self.c * np.exp( -0.5 * v.T @ self.Sigma @ v )

    def gradient(self, point):
        '''
        Gradient of Gaussian function.
        '''
        v = np.array(point) - self.mu
        return - self.c * np.exp( -0.5 * v.T @ self.Sigma @ v ) * ( self.Sigma @ v )

    def hessian(self, point):
        '''
        Hessian of Gaussian function.
        '''
        v = np.array(point) - self.mu
        return - self.c * np.exp( -v.T @ self.Sigma @ v ) * ( self.Sigma - np.outer( self.Sigma @ v, self.Sigma @ v ) )

class Kernel():
    '''
    Class for kernel functions m(x) of maximum degree 2*d, where m(x) is a vector of (n+d,d) known monomials.
    '''
    def __init__(self, dim=2, **kwargs):

        self._dim = dim
        self._degree = 0
        self._state_symbol = 'x'

        for key in kwargs.keys():
            if key.lower() == "symbol":
                self._state_symbol = kwargs["symbol"]
            if key.lower() == "degree":
                self._degree = kwargs["degree"]
            if key.lower() == "monomials":
                monomials = kwargs["monomials"]
                
                length_monomials = np.array([ len(mon) for mon in monomials ])
                if np.any(length_monomials != self._dim):
                    raise Exception("Invalid monomials.")
                
                types = np.array([ isinstance(mon, (list, tuple)) for mon in monomials ])
                exponent_types = np.array([ np.all([ isinstance(term, int) for term in mon ]) for mon in monomials ])
                if not np.all(types) or not np.all(exponent_types):
                    raise Exception("Monomials must be a list/tuple of integer exponents.")
                
                self._powers = monomials

        self._symbols = []
        for dim in range(self._dim):
            self._symbols.append( sym.Symbol(self._state_symbol + str(dim+1)) )

        invalid_type = not isinstance(self._degree, (int, list, tuple))

        is_list = isinstance(self._degree, (list,tuple))
        invalid_list = is_list and ( len(self._degree) != self._dim or np.any([ not isinstance(self._degree[i], int) for i in range(self._dim) ]) )
        if invalid_type or invalid_list:
            raise Exception("Degree must be an integer or list of integers of the same size of the input dimension.")
        
        # If degree is an integer, initialize with all possible monomials up to given degree
        if isinstance(self._degree, int):
            self._max_degree = self._degree
            self._num_monomials = comb(self._dim + self._degree, self._degree)

        # If degree is a list of integers, initialize with monomials up to given degree on each dimension
        if isinstance(self._degree, (list,tuple)):
            self._max_degree = sum(self._degree)
            self._num_monomials = np.prod([ comb(i + self._degree[i], self._degree[i]) for i in range(self._dim) ])

        # Generate monomial list and symbolic monomials
        if not hasattr(self, "_powers"):
            self._powers, self._powers_by_degree = generate_monomials( self._dim, self._degree )

        self._degree = [ 0 for k in range(self._dim) ]
        for k in range(self._dim):
            self._degree[k] = max([ power[k] for power in self._powers ])

        self._monomials = generate_monomial_symbols( self._symbols, self._powers )
        self._num_monomials = len(self._monomials)
        self._K = commutation_matrix(self._num_monomials)       # commutation matrix to be used later

        # Symbolic computations
        P = sym.MatrixSymbol('P', self._num_monomials, self._num_monomials).as_explicit()
        self._Psym = sym.Matrix(self._num_monomials, self._num_monomials, lambda i, j: P[min(i,j),max(i,j)])

        self._sym_monomials = sym.Matrix(self._monomials)
        self._sym_jacobian_monomials = self._sym_monomials.jacobian(self._symbols)

        self._hessian_monomials = [ [0 for i in range(self._dim)] for j in range(self._dim) ]
        for i in range(self._dim):
            for j in range(self._dim):
                self._hessian_monomials[i][j] = sym.diff(self._sym_jacobian_monomials[:,j], self._symbols[i])

        # Compute numeric A and N matrices
        self._compute_A()
        self._compute_N()

        # Lambda functions
        self._lambda_monomials = sym.lambdify( list(self._symbols), self._monomials )
        self._lambda_jacobian_monomials = sym.lambdify( list(self._symbols), self._sym_jacobian_monomials )
        self._lambda_hessian_monomials = sym.lambdify( list(self._symbols), self._hessian_monomials )

        '''
        Obtained the formula for the dimension of the inner blocks of the lowerbound matrix 
        by studying its internal structure. Therefore, self._find_partition(), self._get_block_dependencies(), 
        and self.show_structure() will never need to be used upon the Kernel initialization (would be SUPER costly).
        '''
        n = self._dim
        d = self._max_degree

        r = comb(n+d-2,n)
        s = comb(n+d-2,n-1)
        t = comb(n+d-1,n-1)

        self.blk_sizes = ( min(n+1,r), max(n+1,r)-n-1, s, t )
        self.sl_n = slice(           0            , sum(self.blk_sizes[0:1]))
        self.sl_r = slice(sum(self.blk_sizes[0:1]), sum(self.blk_sizes[0:2]))
        self.sl_s = slice(sum(self.blk_sizes[0:2]), sum(self.blk_sizes[0:3]))
        self.sl_t = slice(sum(self.blk_sizes[0:3]), sum(self.blk_sizes[0:4]))

        '''
        Using self.blk_sizes, we can determine the optimal structure of the P matrix to efficiently solve the SDP for finding a valid CLF.
        '''

    def _validate(self, point):
        ''' Validates input data '''
        if not isinstance(point, (list, tuple, np.ndarray)): raise Exception("Input data point is not a numeric array.")
        if isinstance(point, np.ndarray): point = point.tolist()
        return point

    def _init_CLF_opt(self):
        '''  '''
        n = self._dim
        blk_sizes = self._block_sizes
        r = max(n+1, blk_sizes[0])

        Zeros11 = np.zeros((blk_sizes[0],blk_sizes[0]))
        Zeros22 = np.zeros((blk_sizes[1],blk_sizes[1]))
        Zeros33 = np.zeros((blk_sizes[2],blk_sizes[2]))

        Zeros12 = np.zeros((blk_sizes[0],blk_sizes[1]))
        Zeros13 = np.zeros((blk_sizes[0],blk_sizes[2]))
        Zeros23 = np.zeros((blk_sizes[1],blk_sizes[2]))

        Pnom_var = cp.Parameter( (self._num_monomials, self._num_monomials), symmetric=True )
        Pnn_var = cp.Variable( (n+1,n+1), symmetric=True )
        if r > n+1:
            P1r_var = cp.Variable( (n+1,r-n-1) )
            Prr_var = cp.Variable( (r-n-1, r-n-1), symmetric=True )

            Z = np.zeros((n+1,r-n-1))
            Znn = np.zeros((n+1,n+1))
            Zrr = np.zeros((r-n-1,r-n-1))

            P11_var = cp.bmat([[ Pnn_var ,  Z  ], 
                               [   Z.T   , Zrr ]])
            Pbar11_var = cp.bmat([[   Znn     , P1r_var ], 
                                  [ P1r_var.T , Prr_var ]])
        else:
            P11_var = Pnn_var
            Pbar11_var = np.zeros((n+1,n+1))

        P22_var = cp.Variable( (blk_sizes[1],blk_sizes[1]), symmetric=True )
        P33_var = cp.Variable( (blk_sizes[2],blk_sizes[2]), symmetric=True )
        P12_var = cp.Variable( (blk_sizes[0],blk_sizes[1]) )
        P13_var = cp.Variable( (blk_sizes[0],blk_sizes[2]) )
        P23_var = cp.Variable( (blk_sizes[1],blk_sizes[2]) )

        Pl_var = cp.bmat([ [P11_var  , Zeros12  , Zeros13 ], 
                   [Zeros12.T, P22_var  , P23_var ],
                   [Zeros13.T, P23_var.T, P33_var ] ])

        Pr_var = cp.bmat([ [Pbar11_var, P12_var  , P13_var ], 
                        [P12_var.T , Zeros22  , Zeros23 ],
                        [P13_var.T , Zeros23.T, Zeros33 ] ])
        
        P_var = Pl_var + Pr_var

        cost = cp.norm( P_var - Pnom_var )
        constraints = [ Pl_var >> 0 ]
        constraints += [ lyap(self.Asum2.T, P_var) == 0 ]
        constraints += [ P12_var == 0, P13_var == 0 ]
        if r > n+1: constraints += [ P1r_var == 0 , Prr_var >> 0 ]
        # constraints += [ cp.lambda_max(P_var) <= self.max_eigen_P ]

        self.clf_prob = cp.Problem( cp.Minimize(cost), constraints )

    def _compute_A(self):
        '''
        Computes numeric A matrices.
        '''
        self.A = []
        jacobian_columns = self._sym_jacobian_monomials.T.tolist()
        for dim in range(self._dim):
            Ak = np.zeros([self._num_monomials, self._num_monomials])
            jacobian_column = jacobian_columns[dim]
            for i in range(len(jacobian_column)):
                for j in range(1, self._num_monomials):
                    if len(jacobian_column[i].free_symbols) == 0:
                        if jacobian_column[i] == 0:
                            break
                        else:
                            Ak[i,0] = jacobian_column[i]
                    else:
                        monom_i = jacobian_column[i].as_poly(self._symbols).monoms()
                        monom_j = self._monomials[j].as_poly(self._symbols).monoms()
                        if monom_i[0] == monom_j[0]:
                            Ak[i,j] = jacobian_column[i].as_poly().coeffs()[0]
            self.A.append( Ak )

        self.Asum = sum(self.A)
        self.Asum2 = self.Asum @ self.Asum

    def _compute_N(self):
        '''
        Compute the component matrices of the Jacobian transpose null-space.
        '''
        n = self._dim
        p = self._num_monomials
        I_p = np.eye(p)
        M = np.zeros([n*p**2, p**2])
        for k in range(n):
            A = self.A[k]
            M[k*(p**2):(k+1)*(p**2),:] = np.kron(I_p, A.T) + np.kron(A.T, I_p) @ self._K
            # M[k*(p**2):(k+1)*(p**2),:] = np.kron(I_p, A.T) + np.kron(A.T, I_p)
        # M[n*(p**2):(n+1)*(p**2),:] = self._K - np.eye(p**2)

        from scipy.linalg import null_space
        solutions = null_space(M)

        self.N = []
        for k in range(solutions.shape[1]):
            self.N.append( mat( solutions[:,k] ) )

    def _find_partition(self, Msym):
        ''' 
        Assuming a symmetric input matrix Msym with a structure of the type:
        [ Msym_11    Msym_12
          Msym_12.T     0    ], 
        returns the size of the largest symmetric block (Msym_11).
        '''
        for i in range(self._num_monomials):
            if Msym[i:,i:] == sym.zeros(self._num_monomials-i, self._num_monomials-i):
                if i == 0: return self._num_monomials
                else: return i

    def _get_block_dependencies(self, Msym: sym.Matrix, *slices: list[slice]):
        '''
        Assume a symmetric symbolic input matrix Msym with a block structure of the type:
        Msym = [ Msym_11    Msym_12 ... Msym_1N
                 Msym_12.T  Msym_22 ... Msym_2N
                     .         .           .
                 Msym_1N.T  Msym_N2 ... Msym_NN ], 
        where the slices list fully determine the shape of each block.
        Assuming that Msym is a linear operator of Psym, with the same block structure 
        Psym = [ Psym_11    Psym_12 ... Psym_1N
                 Psym_12.T  Psym_22 ... Psym_2N
                     .         .           .
                 Psym_1N.T  Psym_N2 ... Psym_NN ], 
        this method computes the dependencies of each block Msym_ij on the corresponding blocks Psym_ij.
        '''
        num_slices = len(slices)
        Msym_slices_symbols = [ [ None for _ in range(num_slices) ] for _ in range(num_slices) ]
        Psym_slices_symbols = [ [ None for _ in range(num_slices) ] for _ in range(num_slices) ]

        for i, j in itertools.product(range(num_slices), range(num_slices)):

            sl1_i, sl1_j = slices[i], slices[j]
            Psym_slices_symbols[i][j] = self._Psym[sl1_i, sl1_j].atoms(sym.matrices.expressions.matexpr.MatrixElement)
            Msym_slices_symbols[i][j] = Msym[sl1_i, sl1_j].atoms(sym.matrices.expressions.matexpr.MatrixElement)

        # From here, blocks are constructed
        M_dep = [ [ set() for _ in range(num_slices) ] for _ in range(num_slices) ]
        M_dep_summ = [ [ [] for _ in range(num_slices) ] for _ in range(num_slices) ]
        for i, j in itertools.product(range(num_slices), range(num_slices)):
            curr_Msym_slice = Msym_slices_symbols[i][j]
            
            for m, n in itertools.product(range(num_slices), range(num_slices)):
                curr_Psym_slice = Psym_slices_symbols[m][n]

                common_symbols = curr_Psym_slice & curr_Msym_slice
                M_dep[i][j] = M_dep[i][j] | common_symbols

                if len( common_symbols ) > 0 and m <= n:
                    M_dep_summ[i][j] += [(m+1,n+1)]

        return M_dep, M_dep_summ, Psym_slices_symbols

    def show_structure(self, Msym):
        '''
        Computes the structure of the matrices:
        (As²).T P + P As² + 2 As.T P As
        '''
        Lsym = lyap(self.Asum2.T, self._Psym)
        Lsym_blksize = self._find_partition(Lsym)

        Rsym = 2 * self.Asum.T @ self._Psym @ self.Asum
        Rsym_blksize = self._find_partition(Rsym)

        sl1 = slice(0, Lsym_blksize)
        sl2 = slice(Lsym_blksize, Rsym_blksize)
        sl3 = slice(Rsym_blksize, self._num_monomials)

        M_deps, M_deps_summ, P_struc = self._get_block_dependencies(Msym, sl1, sl2, sl3)

        return M_deps, M_deps_summ, P_struc, (Lsym_blksize, Rsym_blksize - Lsym_blksize, self._num_monomials - Rsym_blksize )

    def get_left_lowerbound(self, shape_matrix):
        ''' Compute the left part L(P) of the matrix lowerbound on the maximum eigenvalue of the Hessian matrix, M(P) = L(P) + R(P) '''
        return lyap(self.Asum2.T, shape_matrix)

    def get_right_lowerbound(self, shape_matrix):
        ''' Compute the left part L(P) of the matrix lowerbound on the  maximum eigenvalue of the Hessian matrix, M(P) = L(P) + R(P) '''
        return 2 * self.Asum.T @ shape_matrix @ self.Asum

    def get_lowerbound(self, shape_matrix):
        ''' Compute the matrix for the lowerbound on the maximum eigenvalue of the Hessian matrix '''
        L = self.get_left_lowerbound(shape_matrix)
        R = self.get_right_lowerbound(shape_matrix)
        return L + R

    def get_constrained_shape(self, shape_matrix):
        ''' Get constrained shape matrix '''

        blk_sizes = self.blk_sizes
        sl_n, sl_r, sl_s, sl_t = self.sl_n, self.sl_r, self.sl_s, self.sl_t

        Zeros_nr = np.zeros((blk_sizes[0],blk_sizes[1]))
        Zeros_ns = np.zeros((blk_sizes[0],blk_sizes[2]))
        Zeros_nt = np.zeros((blk_sizes[0],blk_sizes[3]))

        Zeros_rs = np.zeros((blk_sizes[1],blk_sizes[2]))
        Zeros_rt = np.zeros((blk_sizes[1],blk_sizes[3]))

        if blk_sizes[0] == 0 and blk_sizes[1] == 0:
            constr_SHAPE = cp.bmat([ [ shape_matrix[sl_s,sl_s]  , shape_matrix[sl_s,sl_t] ],
                                     [ shape_matrix[sl_s,sl_t].T, shape_matrix[sl_t,sl_t] ] ])
        elif blk_sizes[1] == 0:
            constr_SHAPE = cp.bmat([ [ shape_matrix[sl_n,sl_n],     Zeros_ns      ,     Zeros_nt     ], 
                                     [    Zeros_ns.T   , shape_matrix[sl_s,sl_s]  , shape_matrix[sl_s,sl_t] ],
                                     [    Zeros_nt.T   , shape_matrix[sl_s,sl_t].T, shape_matrix[sl_t,sl_t] ] ])
        else:
            constr_SHAPE = cp.bmat([ [ shape_matrix[sl_n,sl_n],     Zeros_nr    ,     Zeros_ns      ,     Zeros_nt     ], 
                                     [    Zeros_nr.T   , shape_matrix[sl_r,sl_r],     Zeros_rs      ,     Zeros_rt     ],
                                     [    Zeros_ns.T   ,     Zeros_rs.T  , shape_matrix[sl_s,sl_s]  , shape_matrix[sl_s,sl_t] ],
                                     [    Zeros_nt.T   ,     Zeros_rt.T  , shape_matrix[sl_s,sl_t].T, shape_matrix[sl_t,sl_t] ] ])
        return constr_SHAPE

    def function(self, point):
        ''' Compute kernel function '''
        return np.array(self._lambda_monomials(*self._validate(point)))

    def jacobian(self, point):
        ''' Compute kernel Jacobian '''
        return np.array(self._lambda_jacobian_monomials(*self._validate(point)))

    def get_A_matrices(self):
        '''
        Return the A matrices.
        '''
        return self.A

    def get_N_matrices(self):
        '''
        Return the N matrices.
        '''
        return self.N

    def get_constraints(self, point):
        '''
        Returns kernel constraints
        '''
        from common import kernel_constraints
        F, _ = kernel_constraints( point, self._powers_by_degree )
        return F

    def get_matrix_constraints(self):
        '''
        Returns kernel constraints
        '''
        from common import kernel_constraints
        _, matrices = kernel_constraints( np.zeros(self._num_monomials), self._powers_by_degree )
        return matrices

    def kernel2state(self, kernel_point):
        '''
        This function converts from kernel space to state space, if given point is valid.
        '''
        # if not self.is_in_kernel_space(kernel_point):
        #     raise Exception("Given point is not in the kernel image.")

        return np.flip(kernel_point[1:self._dim+1])

    def is_in_kernel_space(self, point):
        '''
        This function checks whether a given point is inside the kernel space.
        '''
        if len(point) != self.kernel_dim:
            raise Exception("Point must be of the kernel dimension.")

        from common import kernel_constraints
        F, _ = kernel_constraints( point, self._powers_by_degree )
        if np.linalg.norm(F) < 0.00000001:
            return True
        else:
            return False

    def __eq__(self, other):
        '''
        Determines if two kernels are the same.
        '''
        return np.all( self._powers == other._powers )

    def __str__(self):
        ''' Prints kernel '''
        variables = str(self._symbols)
        kernel = str(self._monomials)
        text = f"m: R^{self._dim} --> R^{self._num_monomials}\nPolynomial kernel map of max. degree {self._degree} on variables " + variables + "\nm(x) = " + kernel
        return text

@dataclass
class LeadingShape:
    ''' Data class for a leading shape (to be used as an approximation tool) '''
    shape: np.ndarray
    bound: str = ''
    approximate: bool = False

@dataclass
class MultiPoly:
    '''
    Data class representing multivariable polynomials. 
    To be used as a tool for the Kernel, KernelLinear and KernelQuadratic classes,
    simply to represent polynomial data and perform operations between polynomials.
    '''
    kernel: list[tuple[int]]
    coeffs: list = None
    data_type: None = None # scalar, vector or matrix

    def __post_init__(self):
        '''  Post constructor '''

        if self.coeffs == None:
            self.coeffs = [ None for _ in self.kernel ]

        if isinstance(self.coeffs[0], list):
            self.coeffs = [ np.array(coeff) for coeff in self.coeffs ]

        # Collect data types from each coefficient
        data_types = []
        for coeff in self.coeffs:
            if isinstance( coeff, (int, float, sym.Expr)):
                data_types.append("scalar")
            elif isinstance( coeff, np.ndarray):
                if coeff.ndim == 1:
                    data_types.append("vector")
                if coeff.ndim == 2:
                    data_types.append("matrix")
            else:
                data_types.append(None)
        
        # Checks if coefficients are all of the same data type (necessary for performing operations)
        if data_types.count(data_types[0]) == len(data_types):
            self.data_type = data_types[0]
        else:
            raise Exception("Coefficients are not the same data type.")

        self._sort_kernel()

    def _sort_kernel(self):
        ''' Reordering of passed monomials, according to: total degree/ordering of variables '''

        def degree_order(zipped_item):
            mon = zipped_item[0]
            # coeff = zipped_item[1]
            return sum(mon)

        def pos_order(zipped_item):
            mon = zipped_item[0]
            # coeff = zipped_item[1]
            return sum([ mon[dim]*(2**dim) for dim in range(len(mon)) ])

        zipped = list(zip(self.kernel, self.coeffs))
        zipped.sort(key=degree_order)

        powers_by_degree = {}
        coeffs_by_degree = {}
        for key, zipped_group in itertools.groupby(zipped, degree_order):
            powers_by_degree[key] = []
            coeffs_by_degree[key] = []
            for ele in zipped_group:
                powers_by_degree[key].append( ele[0] )
                coeffs_by_degree[key].append( ele[1] )

            k_th_zipped = list(zip( powers_by_degree[key], coeffs_by_degree[key] ))
            k_th_zipped.sort(key=pos_order)

            powers_by_degree[key] = [ ele[0] for ele in k_th_zipped ]
            coeffs_by_degree[key] = [ ele[1] for ele in k_th_zipped ]
        
        powers, coeffs = [], []
        for key in powers_by_degree.keys():
            powers += powers_by_degree[key]
            coeffs += coeffs_by_degree[key]

        self.kernel = powers
        self.coeffs = coeffs

    def _verify_op(poly1, poly2):
        ''' Checks if aritmetic operations can be executed '''

        type1 = np.array(poly1.kernel).dtype
        type2 = np.array(poly2.kernel).dtype
        if (type1, type2) != ('int','int'):
            raise Exception("Monomial exponents must be integers.")

        if poly1.data_type != poly2.data_type:
            raise Exception("Cannot perform aritmetic operations btw polynomials of different data types.")
        
        if poly1.data_type == "vector":
            if len(poly1.coeffs[0]) != len(poly2.coeffs[0]):
                raise Exception("Cannot perform aritmetic operations btw vector polynomials of different dimensions.")

        if poly1.data_type == "matrix":
            if poly1.coeffs[0].shape != poly2.coeffs[0].shape:
                raise Exception("Cannot perform aritmetic operations btw matrix polynomials of different dimensions.")

    def _addition(poly1, poly2, op):
        '''
        Add/subtract two instances of multipoly.
        op = +1 sums, op = -1 subtracts
        '''
        MultiPoly._verify_op(poly1, poly2)

        res_kernel = list( set(poly1.kernel).union(set(poly2.kernel)) )
        
        if poly1.data_type == "scalar":
            res_coeffs = [ 0.0 for _ in res_kernel ]
        if poly1.data_type in ("vector", "matrix"):
            res_coeffs = [ np.zeros(poly1.coeffs[0].shape) for _ in res_kernel ]

        for k, mon in enumerate(res_kernel):

            if mon in poly1.kernel:
                i = poly1.kernel.index( mon )
                res_coeffs[k] += poly1.coeffs[i]

            if mon in poly2.kernel:
                i = poly2.kernel.index( mon )
                if op > 0: 
                    res_coeffs[k] += poly2.coeffs[i]
                else: 
                    res_coeffs[k] -= poly2.coeffs[i]

        return MultiPoly(kernel=res_kernel, coeffs=res_coeffs)

    def _add(poly1, poly2):
        ''' Add two instances of multipoly '''
        return MultiPoly._addition(poly1, poly2, +1)

    def _sub(poly1, poly2):
        ''' Subtract two instances of multipoly '''
        return MultiPoly._addition(poly1, poly2, -1)

    def _multiply(poly1, poly2, op):
        '''
        Polynomial multiplication (term by term or matrix-like).
        op = +1 for ter-by-term, op = -1 for matrix-like
        '''
        MultiPoly._verify_op(poly1, poly2)

        # Initialization of product kernel
        res_kernel = []
        for mon1, mon2 in itertools.product( poly1.kernel, poly2.kernel ):
            mon = tuple([ int(dim1)+int(dim2) for dim1, dim2 in zip(mon1,mon2) ])
            if mon not in res_kernel:
                res_kernel.append(mon)

        # Initialization of product coefficients
        res_coeffs = []
        data_sample = poly1.coeffs[0]
        if isinstance(data_sample, (int, float, sym.Symbol)):
            res_coeffs = [ 0.0 for _ in range(len(res_kernel)) ]
        if isinstance(data_sample, (np.ndarray)):
            res_coeffs = [ np.zeros(data_sample.shape) for _ in range(len(res_kernel)) ]
            # if data_sample.ndim == 2: 
            #     multiply_as_matrix = True

        # Populate coefficients
        zipped1 = zip(poly1.kernel, poly1.coeffs)
        zipped2 = zip(poly2.kernel, poly2.coeffs)
        for (z1, z2) in itertools.product( zipped1, zipped2 ):

            mon1, coeff1 = z1[0], z1[1]
            mon2, coeff2 = z2[0], z2[1]

            mon = tuple([ int(dim1)+int(dim2) for dim1, dim2 in zip(mon1,mon2) ])
            index = res_kernel.index( mon )
            
            if op > 0:
                res_coeffs[index] += coeff1 * coeff2
            else:
                res_coeffs[index] += coeff1 @ coeff2

        return MultiPoly(kernel=res_kernel, coeffs=res_coeffs)

    def _mul(poly1, poly2):
        ''' Term by term polynomial multiplication '''
        return MultiPoly._multiply(poly1, poly2, +1)

    def _matmul(poly1, poly2):
        ''' Matrix polynomial multiplication '''

        if poly1.data_type == "scalar":
            return NotImplemented
        return MultiPoly._multiply(poly1, poly2, -1)

    def _operator_fallbacks(operation, op_name):
        ''' Implementation of forward, reverse and inplace operations for MultiPoly '''

        def forward(a, b):
            ''' Implementation of forward op '''

            if isinstance(b, MultiPoly):
                return operation(a, b)
            if isinstance(b, (int, float)):
                return operation(a, MultiPoly(kernel=[(0,0)], coeffs=[b]) )
            else:
                return NotImplemented
        forward.__name__ = '__' + op_name.__name__ + '__'
        forward.__doc__ = operation.__doc__

        def reverse(b, a):
            ''' Implementation of reverse op '''

            if isinstance(a, MultiPoly):
                return operation(a, b)
            elif isinstance(a, (int, float)):
                return operation(MultiPoly(kernel=[(0,0)], coeffs=[a]), b )
            else:
                return NotImplemented
        reverse.__name__ = '__r' + op_name.__name__ + '__'
        reverse.__doc__ = operation.__doc__

        def inplace(a, b):
            ''' Implementation of inplace op '''
            
            if isinstance(a, MultiPoly):
                return operation(a, b)
            elif isinstance(a, (int, float)):
                return operation(a, MultiPoly(kernel=[(0,0)], coeffs=[b]) )
            else:
                return NotImplemented
        inplace.__name__ = '__i' + op_name.__name__ + '__'
        inplace.__doc__ = operation.__doc__

        return forward, reverse, inplace

    __add__, __radd__, __iadd__ = _operator_fallbacks(_add, operator.add)
    __sub__, __rsub__, __isub__ = _operator_fallbacks(_sub, operator.sub)
    __mul__, __rmul__, __imul__ = _operator_fallbacks(_mul, operator.mul)
    __matmul__, __rmatmul__, __imatmul__ = _operator_fallbacks(_matmul, operator.matmul)

    def __repr__(self):
        ''' Representation of MultiPoly '''
        poly_repr = f"{self.data_type.capitalize()} poly on x:\n"
        for coeff, power in zip(self.coeffs, self.kernel):
            if isinstance(coeff, (int, float, sym.Expr)):
                if coeff > 0: sign_text = "+ "
                else: sign_text = "- "
                poly_repr += sign_text + f"{abs(coeff):.3f}*x^{power} "
            else:
                poly_repr +=  f"{coeff}*x^{power} + "
        return poly_repr

    def __str__(self):
        ''' Printing for MultiPoly '''
        return self.__repr__()

class KernelLinear(Function):
    '''
    Class for multidimensional polynomial functions of the type f(x) = ∑ c_i [x]^a_i, 
    where the a_i = [ i_1, i_2, ... i_n ] are multidimensional exponents defining monomials of max. degree (i_1 + i_2 + i_n):
    [x]^a_i = x_1^(i_1) x_2^(i_2) ... x_n^(i_n)

    They use a given polynomial kernel and have constant coefficients c_i, which can be scalars, vectors or matrices.
    In case of matrix coefficients, KernelLinear represents a class of polynomial matrices on the given kernel.
    ''' 
    def __init__(self, **kwargs):

        self._dim = 2
        super().__init__(**kwargs)

    def _fun(self, x, coeffs):
        ''' Returns the function value using given coefficients '''
        m = self.kernel.function(x)
        return sum([ coeffs[k] * m[k] for k in range(self.kernel_dim) ])

    def _function(self, x):
        ''' Returns function using self configuration '''
        return self._fun(x, self.coeffs )

    def _gradient(self, x):
        ''' Returns gradient using self configuration '''
        logging.warning("Currently not implemented.")

    def _hessian(self, x):
        ''' Returns hessian matrix using self configuration '''
        logging.warning("Currently not implemented.")

    def _initialize(self, kernel):
        '''
        Given a kernel, correctly initialize function.
        '''
        self.kernel = kernel
        self._dim = self.kernel._dim
        self.kernel_dim = self.kernel._num_monomials
        self.kernel_matrices = self.kernel.get_A_matrices()
        self.coeffs = [ 0.0 for _ in range(self.kernel_dim) ]

        self._compute_sos()
        self._compute_sos_index()

    def _compute_sos(self):
        '''
        Function for computing the corresponding polynomial SOS representation.
        Returns: (i) the needed polynomial kernel for SOS-factorization (check)
                (ii) the rule for generating a corresponding shape matrix, from the coefficients
        '''
        self._sos_monomials = []
        for mon in self.kernel._powers:
            possible_curr_combinations = set([ tuple(np.array(mon1)+np.array(mon2)) for mon1,mon2 in itertools.combinations(self._sos_monomials, 2) ])

            if mon in possible_curr_combinations: 
                continue

            if len(possible_curr_combinations) == 0: 
                self._sos_monomials.append(mon)
                continue

            # If mon is not on possible with current combinations, check if its possible to create it from them...
            possibilities = []

            # If all exponents of mon are even, it can be created from 
            if np.all([ exp % 2 == 0 for exp in mon ]):
                possibilities.append( tuple([int(exp/2) for exp in mon]) )

            # Checks if mon can be created from the combination of monomials already in self._sos_monomials and another
            for sos_mon in self._sos_monomials:
                pos = np.array(mon) - np.array(sos_mon)
                if np.all(pos >= 0): 
                    possibilities.append( tuple([ int(exp) for exp in pos ]) )

            index = np.argmin([ np.linalg.norm(pos) for pos in possibilities ])
            new_sos_mon = possibilities[index]
            if new_sos_mon not in self._sos_monomials:
                self._sos_monomials.append(new_sos_mon)

    def _compute_sos_index(self):
        '''
        Computes the index matrix representing the rule for placing the coefficients in the correct places on the 
        shape matrix of the SOS representation. Algorithm gives preference for putting the elements of coeffs 
        closer to the main diagonal of the SOS matrix.
        '''     
        sos_kernel_dim = len(self._sos_monomials)
        self._index_matrix = -np.ones([sos_kernel_dim, sos_kernel_dim], dtype='int')

        for k in range(self.kernel_dim):

            mon = self.kernel._powers[k]

            # Checks the possible (i,j) locations on SOS matrix where the monomial can be put
            possible_places = []
            for (i,j) in itertools.product(range(sos_kernel_dim),range(sos_kernel_dim)):
                if i > j: continue
                sos_mon_i, sos_mon_j = np.array(self._sos_monomials[i]), np.array(self._sos_monomials[j])

                if mon == tuple(sum([sos_mon_i, sos_mon_j])):
                    possible_places.append( (i,j) )

            # From these, chooses the place closest to SOS matrix diagonal
            distances_from_diag = np.array([ np.abs(place[0] - place[1]) for place in possible_places ])
            i,j = possible_places[np.argmin(distances_from_diag)]

            self._index_matrix[i,j] = k

    def set_coefficients(self, coeffs):
        ''' Setting method for coefficients '''

        if not isinstance(coeffs, (list, tuple, np.ndarray)):
            raise Exception("Coefficients must be array-like.")

        if len(coeffs) != self.kernel_dim:
            raise Exception("Number of coefficients must be the same as the kernel dimension.")

        # Scalar-valued function
        is_scalar = are_all_type(coeffs, (int,float))
        if is_scalar:
            self._func_type = "scalar"
            self._output_dim = 1

        # Vector/Matrix-valued function
        is_multidim = are_all_type(coeffs, (list, tuple, np.ndarray))
        if is_multidim:
            ndims = np.array([ np.array(coeff).ndim for coeff in coeffs ])
            
            if not ndims.tolist().count(ndims[0]) == len(ndims):
                raise Exception("Passed arrays have different number of dimensions.")
            
            if np.all(ndims == 1):
                self._func_type = "vector"
                self._output_dim = len(coeffs[0])
            elif np.all(ndims == 2):
                self._func_type = "matrix"
                self._output_dim = coeffs[0].shape
            else:
                raise Exception("KernelLinear class only supports scalar, vectors or matrices as coefficients.")

        self.coeffs = coeffs

        # If function is scalar, load the sos_shape_matrix corresponding to the coefficients 
        if self._func_type == "scalar":
            self._sos_shape_matrix = np.array( self.get_sos_shape(self.coeffs) )

    def set_params(self, **kwargs):
        ''' Sets function parameters '''

        super().set_params(**kwargs)

        keys = [ key.lower() for key in kwargs.keys() ] 

        if "kernel" in keys:
            if type(kwargs["kernel"]) != Kernel:
                raise Exception("Argument must be a valid Kernel function.")
            self._initialize( kwargs["kernel"] )

        if "degree" in keys:
            if "dim" in keys: self._dim = kwargs["dim"]
            else: print("Kernel dimension was not specified. Initializing with new Kernel of n = 2.")
            self._initialize( Kernel(dim=self._dim, degree=kwargs["degree"]) )

        for key in keys:
            if key in ["kernel", "degree"]: # Already dealt with
                continue
            if key == "coeffs":
                self.set_coefficients( kwargs["coeffs"] )
                continue

    def get_sos_shape(self, coeffs):
        '''
        Using the index matrix, returns the SOS shape matrix correctly populated by the coefficients.
        '''
        if len(coeffs) != self.kernel_dim:
            raise Exception("The number of coefficients must be equal to the kernel dimension.")
        
        sos_kernel_dim = len(self._sos_monomials)
        shape_matrix = np.zeros([sos_kernel_dim, sos_kernel_dim]).tolist()

        for (i,j) in itertools.product(range(sos_kernel_dim),range(sos_kernel_dim)):
            if i > j: continue

            k = self._index_matrix[i,j]
            if k >= 0:
                if i == j:
                    shape_matrix[i][j] = coeffs[k]
                else:
                    shape_matrix[i][j] = 0.5 * coeffs[k]
                    shape_matrix[j][i] = 0.5 * coeffs[k]

        return shape_matrix
        
    def determinant(self):
        ''' Compute the determinant polynomial if the coefficients are square matrices (by Laplace expansion) '''
        
        if self.data_type != "matrix":
            raise Exception("Not a polynomial matrix. Cannot compute determinant.")
        if self.coeffs[0].shape[0] != self.coeffs[0].shape[1]:
            raise Exception("Cannot compute determinant of a non-square matrix.")

        if self._dim != 2:
            pass

    def get_kernel(self):
        ''' Returns the monomial basis vector '''
        return self.kernel

    @classmethod
    def from_poly(cls, poly: MultiPoly):
        ''' Constructor for creating KernelLinear from a MultiPoly '''

        n = len(poly.kernel[0])
        kernel = Kernel(dim=n, monomials=poly.kernel)

        return cls(kernel=kernel, coeffs=poly.coeffs)

    def __str__(self):
        ''' Prints KernelLinear '''

        if hasattr(self, "_func_type"):
            text = f"{self._func_type.capitalize()}-valued polynomial f: R^{self._dim} --> R^{self._output_dim} created from "
        else:
            text = f"Polynomial f: R^{self._dim} --> R^? created from "
        kernel_text = f"polynomial kernel map of max. degree {self.kernel._degree} on variables {self.kernel._symbols}"
        return text + kernel_text

class KernelQuadratic(Function):
    '''
    Class for kernel quadratic functions of the type f(x) = m(x)' F m(x) - C for a given kernel m(x), where:
    F is a p.s.d. matrix and C is an arbitrary constant.
    '''
    def __init__(self, **kwargs):

        self.constant = 0.0

        self.cost = 0.0
        self.constraints = []
        self.force_coords = False
        self.force_gradients = False
        self.last_opt_results = None
        self.max_eigenvalue = 10.0

        super().__init__(**kwargs)
        self.reset_optimization()

    def __str__(self):
        return "Polynominal kernel-based function h(x) = ½ ( k(x)' M k(x) - c )"

    def _initialize(self, kernel):
        '''
        Given a kernel, correctly initialize function.
        '''
        self.kernel = kernel
        self._dim = self.kernel._dim
        self.kernel_dim = self.kernel._num_monomials
        self.kernel_matrices = self.kernel.get_A_matrices()
        self.shape_matrix = None

        self.param_dim = int(self.kernel_dim*(self.kernel_dim + 1)/2)
        self.dynamics = Integrator( np.zeros(self.param_dim), np.zeros(self.param_dim) )

        self.SHAPE = cp.Variable( (self.kernel_dim,self.kernel_dim), symmetric=True )

        self.reset_optimization()
            
    def _gradient_const_matrices(self, shape_matrix):
        ''' Compute constant matrices composing the elements of the gradient '''

        grad_list = [ Ai.T @ shape_matrix for Ai in self.kernel_matrices ]
        return grad_list

    def _hessian_const_matrices(self, shape_matrix):
        ''' Compute constant matrices composing the elements of the Hessian '''

        H_list = [ [ ( Ai.T @ shape_matrix + shape_matrix @ Ai ) @ Aj for Aj in self.kernel_matrices ] for Ai in self.kernel_matrices ]
        return H_list

    def _fun(self, x, shape_matrix):
        ''' Returns the function value '''
        m = self.kernel.function(x)
        return 0.5 * m.T @ shape_matrix @ m - self.constant

    def _grad(self, x, shape_matrix):
        ''' Gradient vector as a function of the state and shape matrix '''

        m = self.kernel.function(x)
        Jm = self.kernel.jacobian(x)

        return Jm.T @ shape_matrix @ m

    def _hess(self, x, shape_matrix):
        ''' Hessian matrix as a function of the state and shape matrix '''

        m = self.kernel.function(x)
        H_list = self._hessian_const_matrices(shape_matrix)

        return [ [ m.T @ H_list[i][j] @ m for j in range(self._dim) ] for i in range(self._dim) ]

    def _function(self, point):
        ''' Returns function using self configuration '''
        return self._fun(point, self.shape_matrix)

    def _gradient(self, point):
        ''' Returns gradient using self configuration '''
        return self._grad(point, self.shape_matrix)

    def _hessian(self, point):
        ''' Returns hessian using self configuration '''
        return np.array(self._hess( point, self.shape_matrix ))

    def _hessian_quadratic_form(self, x, shape_matrix, v):
        ''' Computes the quadratic form v' H v with hessian matrix H and vector v'''

        m = self.kernel.function(x)
        H_list = self._hessian_const_matrices(shape_matrix)
        M = sum([ H_list[i][j] * v[i] * v[j] for (i,j) in itertools.product(range(self._dim), range(self._dim)) ])
        return m.T @ M @ m

    def _SOSconvex_matrix(self, shape_matrix):
        ''' Returns SOS convexity matrix '''
        return np.block([[ Ai.T @ shape_matrix @ Aj + Aj.T @ Ai.T @ shape_matrix for Aj in self.kernel_matrices ] for Ai in self.kernel_matrices ])

    def reset_optimization(self):
        ''' Reset optimization problem '''
        
        self.cost = 0.0
        self.constraints = []
        self.add_psd_constraint()

    def get_kernel(self):
        ''' Returns the monomial basis vector '''
        return self.kernel

    def get_shape(self):
        ''' Returns the polynomial coefficients '''
        return self.shape_matrix

    def get_curvature(self, point):
        '''
        For testing only. Only works in 2D
        '''
        if self._dim != 2:
            raise Exception("Not intended to work with dimensions higher than 2.")

        grad = self.gradient(point)
        grad_norm = np.linalg.norm(grad)
        normalized_grad = grad/grad_norm
        z = rot2D(np.pi/2) @ normalized_grad
        Hessian = self.hessian(point)

        return z.T @ Hessian @ z / grad_norm

    def is_SOS_convex(self, verbose=False) -> bool:
        ''' Returns True if the function is SOS convex '''

        sos_convex = False
        SOS_eigs = np.linalg.eigvals( self._SOSconvex_matrix(self.shape_matrix) )
        if np.all(SOS_eigs >= 0.0): sos_convex = True

        if verbose:
            if sos_convex: print(f"{self} is SOS convex.")
            else: print(f"{self} is not SOS convex, with negative eigenvalues = {SOS_eigs[SOS_eigs < 0.0]}")

        return sos_convex

    def is_bounded_by(self, shape_bound: np.ndarray, verbose=False, threshold=1e-3) -> bool:
        ''' Returns type of bound for shape_bound:
            'lower' if SHAPE >> shape_bound, 
            'upper' if SHAPE << shape_bound, 
            None if shape_bound is not a bound
        '''
        bound = None
        lowerbounded, upperbounded = False, False

        lowerbound_eigs = np.linalg.eigvals( self.shape_matrix - np.array(shape_bound) )
        if np.all(lowerbound_eigs >= -threshold): 
            lowerbounded = True
            bound = 'lower'

        upperbound_eigs = np.linalg.eigvals( np.array(shape_bound) - self.shape_matrix )
        if np.all(upperbound_eigs >= -threshold): 
            upperbounded = True
            bound = 'upper'

        if verbose: 
            if lowerbounded: message = f"{self} is lowerbounded by passed shape matrix,"
            else: message = f"{self} is not lowerbounded by passed shape matrix,"
            message += f" with negative eigenvalues = {lowerbound_eigs[lowerbound_eigs < 0.0]}"
            print(message)

            if upperbounded: message = f"{self} is upperbounded by passed shape matrix,"
            else: message = f"{self} is not upperbounded by passed shape matrix,"
            message += f" with negative eigenvalues = {upperbound_eigs[upperbound_eigs < 0.0]}"
            print(message)

        return bound

    def set_shape(self, shape_matrix):
        ''' Setting method for tyhe shape matrix. '''

        # If a one dimensional array was passed, checks if it can be converted to symmetric matrix
        if shape_matrix.ndim == 1:
            roots = np.roots([1, 1, -2*len(shape_matrix)])
            if np.any([ root.is_integer() and root > 0 for root in roots ]):
                shape_matrix = vector2sym(shape_matrix.tolist())
            else:
                raise Exception("Number of coefficients is not compatible with a symmetric matrix.")

        if shape_matrix.shape[0] != shape_matrix.shape[1]:
            raise Exception("Matrix of coefficients must be a square.")
        
        if not np.all( shape_matrix == shape_matrix.T ):
            warnings.warn("Matrix of coefficients is not symmetric. The symmetric part will be used.")

        # if not np.all(np.linalg.eigvals(shape_matrix) >= -1e-5):
        #     raise Exception("Matrix of coefficients must be positive semi-definite.")

        if shape_matrix.shape != (self.kernel_dim, self.kernel_dim):
            raise Exception("Matrix of coefficients doesn't match the kernel dimension.")
        
        self.shape_matrix = 0.5 * ( shape_matrix + shape_matrix.T )
        self.param = sym2vector( self.shape_matrix )
        self.dynamics.set_state(self.param)

    def set_params(self, **kwargs):
        ''' Sets function parameters '''

        super().set_params(**kwargs)

        keys = [ key.lower() for key in kwargs.keys() ] 

        if "kernel" in keys:
            if type(kwargs["kernel"]) != Kernel:
                raise Exception("Argument must be a valid Kernel function.")
            self._initialize( kwargs["kernel"] )

        if "degree" in keys:
            if "dim" in keys: self._dim = kwargs["dim"]
            else: print("Kernel dimension was not specified. Initializing with new Kernel of n = 2.")
            self._initialize( Kernel(dim=self._dim, degree=kwargs["degree"]) )

        if "constant" in keys:
            self.constant = kwargs["constant"]

        for key in keys:

            if key in ["constant", "kernel", "degree"]: # Already dealt with
                continue

            if key == "coefficients":
                self.set_shape( np.array(kwargs["coefficients"]) )
                continue

            if key == "force_coords":
                self.force_coords = kwargs["force_coords"]
                continue

            if key == "force_gradients":
                self.force_gradients = kwargs["force_gradients"]
                continue

            if key == "no_maxima" and kwargs["no_maxima"]:
                self.add_no_maxima_constraint()
                continue

            if key == "points":
                for point_dict in kwargs["points"]:
                    self.add_point_constraints(**point_dict)
                continue

            if key == "centers":
                self.add_center_constraints( kwargs["centers"] )
                continue

            if key == "boundary":
                self.add_boundary_constraints( kwargs["boundary"] )
                continue

            if key == "skeleton":
                self.add_skeleton_constraints( kwargs["skeleton"] )
                continue

            if key == "safe":
                self.add_safe_constraints( kwargs["safe"] )
                continue

            if key == "unsafe":
                self.add_unsafe_constraints( kwargs["unsafe"] )
                continue

            if key == "leading":
                leading = kwargs["leading"]
                if not isinstance(leading, LeadingShape):
                    raise Exception("leading must be of the class LeadingShape()")
                if leading.shape.shape != (self.kernel_dim, self.kernel_dim):
                    raise Exception("Leading shape matrix must be (p x p), where p is the kernel dimension.")
                self.add_leading_constraints(leading)
                continue

        # If fitting conditions are satisfied, fits and sets new, fitted shape
        if type(self.cost) != int and len(self.constraints) > 2 and not np.any(self.shape_matrix):
            fitted_shape = self.fitting()
            self.set_shape(fitted_shape)

    def add_psd_constraint(self):
        ''' Positive semi definite constraint for CVXPY optimization '''
        self.constraints += [ self.SHAPE >> 0 ]
        self.constraints += [ cp.lambda_max(self.SHAPE) <= self.max_eigenvalue ]

    def add_no_maxima_constraint(self): 
        ''' Non-negative definite Hessian constraint for CVXPY optimization. Prevents occurrence of local maxima '''
        self.constraints += [ self.SHAPE @ col == 0 for col in self.kernel.Asum2.T if np.any(col != 0.0) ]

    def add_point_constraints(self, **point):
        '''
        Adds point-like constraints to optimization problem.
        Parameter: point = { "coords": ArrayLike, "level": float >= -self.constant, "gradient": ArrayLike, "curvature" : float }
        '''
        keys = point.keys()

        if "coords" not in keys: raise Exception("Point coordinates must be specified.")
        if "force_coord" not in keys: point["force_coord"] = False
        if "force_gradient" not in keys: point["force_gradient"] = False

        # Define point-level constraints
        if "level" in keys:
            if point["level"] >= -self.constant:
                if self.force_coords or point["force_coord"]:
                    self.constraints += [ self._fun(point["coords"], self.SHAPE) == point["level"] ]
                else:
                    self.cost += ( self._fun(point["coords"], self.SHAPE) - point["level"] )**2

        # Define gradient constraints
        if "gradient" in keys:
            gradient_norm = cp.Variable()
            gradient = np.array(point["gradient"])
            normalized = gradient/np.linalg.norm(gradient)

            self.constraints += [ gradient_norm >= 0 ]
            if self.force_gradients or point["force_gradient"]:
                self.constraints += [ self._grad(point["coords"], self.SHAPE) == gradient_norm * normalized ]
            else:
                self.cost += cp.norm( self._grad(point["coords"], self.SHAPE) - gradient_norm * normalized )

        # Define curvature constraints (2D only)
        if "curvature" in keys:
            if self._dim != 2:
                raise Exception("Error: curvature fitting was not implemented for dimensions > 2. ")
            if "gradient" not in keys:
                raise Exception("Cannot specify a curvature without specifying the gradient.")

            v = rot2D(np.pi/2) @ normalized
            self.cost += ( self._hessian_quadratic_form(point["coords"], self.SHAPE, v) - point["curvature"] )**2

    def add_leading_constraints(self, leading: LeadingShape):
        '''
        Defines a leading function. Can be used as an lower bound, upper bound or as an approximation.
        Parameters: Pleading = (p x p) np.ndarray, where p is the kernel space dimension
                    bound = int (< 0, 0, >0): if zero, no bound occurs. If positive/negative, passed function is a lower/upper bound.
                    approximate = bool: if the function must be approximated.
        '''
        if leading.shape.shape[0] != leading.shape.shape[1]:
            raise Exception("Shape matrix for the bounding function must be square.")
        if leading.shape.shape != (self.kernel_dim, self.kernel_dim):
            raise Exception("Shape matrix and kernel dimensions are incompatible.")
        
        bound_threshold = 0.0

        if leading.bound == "lower": 
            self.constraints += [ self.SHAPE >> leading.shape + bound_threshold*np.eye(self.kernel_dim) ]  # leading is a lowerbound
        elif leading.bound == "upper": 
            self.constraints += [ self.SHAPE << leading.shape - bound_threshold*np.eye(self.kernel_dim) ]  # leading is an upperbound

        if leading.approximate: 
            self.cost += cp.norm( self.SHAPE - leading.shape )          # Pleading will be used as an approximation

    def add_levelset_constraints(self, point_list: list, level: float, contained=False):
        '''
        Adds constraints to set a list of passed points to an specific level set. 
        If contained = True, the points must be completely contained in the level set.
        Parameters: point_list -> list of point coordinates 
                        level  -> value of level set
        Returns: the optimization error.
        '''
        for pt in point_list:
            self.add_point_constraints(coords = pt, level=level)
            if contained: self.constraints.append( self._fun(pt, self.SHAPE) <= level )

    def add_center_constraints(self, point_list: list):
        '''
        Adds constraints to set a list of passed points to the -self.constant level set.
        For CLFs/CBFs, these points will act as minima. 
        '''
        self.add_levelset_constraints(point_list, level=-self.constant)

    def add_safe_constraints(self, point_list: list):
        ''' Adds constraints to guarantee safety of given points '''
        for pt in point_list:
            self.constraints += [ self._fun(pt, self.SHAPE) >= 0.0 ]

    def add_unsafe_constraints(self, point_list: list):
        ''' Adds constraints to guarantee unsafety of given points '''
        for pt in point_list:
            self.constraints += [ self._fun(pt, self.SHAPE) <= 0.0 ]

    def add_boundary_constraints(self, point_list):
        '''
        Adds constraints to set a list of passed points to the 0-level set. Useful for barrier fitting.
        '''
        self.add_levelset_constraints(point_list, 0.0, contained=True)

    def add_continuity_constraints(self, points_sequence, increasing = True):
        '''
        Generates appropriate constraints for smooth variation of the function along a curve.
        The points of the curve are defined by list points_sequence, which are assumed to be ordered.
        '''
        for k in range(len(points_sequence)-1):
            curr_pt = np.array(points_sequence[k])
            next_pt = np.array(points_sequence[k+1])

            inner = (+1 if increasing else -1) * ( next_pt - curr_pt ).T @ self._grad(curr_pt, self.SHAPE)
            self.constraints.append( self._fun(next_pt, self.SHAPE) - self._fun(curr_pt, self.SHAPE) >= inner )
                
    def add_skeleton_constraints(self, skeleton_segments):
        '''
        Generates the appropriate constraints for smooth increasing of the CBF from a center point located on the skeleton curve.
        Parameters: - skeleton_segments is an array with segments, each containing sampled points of the obstacle medial-axis.
                    - the points on each segment are assumed to be ordered: that is, the barrier must grow from one point to the next
        '''
        for segment in skeleton_segments:
            self.add_center_constraints(point_list=segment)
            self.add_continuity_constraints(segment, increasing=True)

    def fitting(self):
        ''' 
        Convex optimization problem for fitting the coefficient matrix to the current cost and constraints.
        Returns: the optimization results.
        '''
        fit_problem = cp.Problem( cp.Minimize( self.cost ), self.constraints )
        fit_problem.solve(verbose=False, max_iters = 100000)

        if "optimal" in fit_problem.status:
            print("Fitting was successful with final cost = " + str(fit_problem.value) + " and message: " + str(fit_problem.status))
            return self.SHAPE.value
        else:
            raise Exception("Problem is " + fit_problem.status + ".")
        
    def update(self, param_ctrl, dt):
        '''
        Integrates and updates parameters
        '''
        self.dynamics.set_control(param_ctrl)
        self.dynamics.actuate(dt)
        self.set_params( coefficients = self.dynamics.get_state() )

class KernelLyapunov(KernelQuadratic):
    '''
    Class for kernel-based Lyapunov functions.
    '''
    def __init__(self, **kwargs):

        kwargs["color"] = mcolors.TABLEAU_COLORS['tab:blue'] # Standard CLF color is blue; however, this can be overwritten
        kwargs["constant"] = 0.0                             # Standard constant for CLF is 0.0 (cannot be overwritten)

        # Initialize the parameters of KernelQuadratic
        super().__init__(**kwargs)

    def __str__(self):
        return "Polynominal kernel-based CLF V(x) = ½ k(x)' P k(x)"

    def set_params(self, **kwargs):
        '''
        Set the parameters of the Kernel Lyapunov function.
        Optional: pass a vector of parameters representing the vectorization of matrix P
        '''
        if "P" in kwargs.keys(): 
            kwargs["coefficients"] = kwargs.pop("P") # Standard name for CLF shape matrix is P

        super().set_params(**kwargs)
        self.P = self.shape_matrix

class KernelBarrier(KernelQuadratic):
    '''
    Class for kernel-based barrier functions.
    '''
    def __init__(self, **kwargs):

        kwargs["color"] = mcolors.TABLEAU_COLORS['tab:red'] # Standard CBF color is red; however, this can be overwritten
        kwargs["constant"] = 0.5                            # Standard constant for CBF is 0.5 (cannot be overwritten)
        
        super().__init__(**kwargs)

    def __str__(self):
        return "Polynominal kernel-based CBF h(x) = ½ ( k(x)' Q k(x) - 1 )"

    def set_params(self, **kwargs):
        ''' Set the parameters of the Kernel Barrier function.
        Optional: pass a vector of parameters representing the vectorization of matrix Q '''

        if "Q" in kwargs.keys(): 
            kwargs["coefficients"] = kwargs.pop("Q") # Standard name for CBF shape matrix is Q

        super().set_params(**kwargs)
        self.Q = self.shape_matrix

    def get_boundary(self):
        ''' Computes the boundary level set '''
        return self.get_levels(levels=[0.0])[0]

class KernelTriplet():
    '''
    Class for kernel-based triplets of: plant, CLF and CBF.
    Defines common algorithms for CLF-CBF compatibility, such as computation of the invariat set, equilibrium points, and optimizations over the invariant set branches.
    The variable self.P is used for online computations with the CLF shape.
    '''
    def __init__(self, **kwargs):
        
        self.plant = None
        self.clf = None
        self.cbf = None
        self.params = {"slack_gain": 1.0, "clf_gain": 1.0, "cbf_gain": 1.0}

        self.limits = [ [-1, +1] for _ in range(2) ]
        self.spacing = 0.1
        self.invariant_color = mcolors.BASE_COLORS["k"]
        self.invariant_complete = True
        self.compatibility_options = { "barrier_sep": 0.1, "min_curvature": 1.1 }
        self.interior_eq_threshold = 1e-1
        self.max_P_eig = 100
        self.invariant_lines_plot = []
        self.plotted_attrs = {}

        self.comp_process_data = {"step": 0, 
                                  "start_time": 0.0, 
                                  "execution_time": 0.0, 
                                  "gui_eventloop_time": 0.3,
                                  "invariant_segs_log": [] }
        
        self.comp_graphics = { "fig": None,
                               "text": None,
                               "clf_artists": [] }

        # Defines main class attributes
        self.set_param(**kwargs)
        
        # Initialize matrix for determinant computation
        self.W = np.empty((self.n,2))

        # Initialize cost and constraints for scipy.optimize computation
        self.cost = 0.0
        self.lowerb_constr = [ 0.0 for col in self.kernel.Asum2.T if np.any(col != 0.0) ]
        self.rem_constr = [ 0.0, 0.0 ]
        self.max_shape_eigenvalue = 100

        # Initialize CVXPY parameters
        self.CVXPY_P = cp.Variable( (self.p,self.p), symmetric=True )
        self.CVXPY_Pnom = cp.Parameter( (self.p,self.p), symmetric=True )
        self.CVXPY_lambdas = []

        self.CVXPY_cost = cp.norm(self.CVXPY_P - self.CVXPY_Pnom)
        self.CVXPY_base_constraints = [ self.CVXPY_P >> 0, cp.lambda_max(self.CVXPY_P) <= self.max_P_eig ]
        self.CVXPY_constraints = self.CVXPY_base_constraints

        # self.CVXPY_family_cost = cp.norm(self.CVXPY_P - self.CVXPY_Pnom)
        # self.CVXPY_family_constraints = [ self.CVXPY_P >> 0, 
        #                                   self.kernel.get_left_lowerbound(self.CVXPY_P) == 0,
        #                                   self.kernel.get_constrained_shape(self.CVXPY_P) == self.CVXPY_P,
        #                                   cp.lambda_max(self.CVXPY_P) <= self.max_P_eig ]
        
        # self.CVXPY_family_problem = cp.Problem( cp.Minimize( self.CVXPY_family_cost ), self.CVXPY_family_constraints )

        # Compute limit lines (obstacle should be completely contained inside the rectangle)
        self.create_limit_lines()

        # Compute CBF boundary
        self.compute_cbf_boundary()

        # Initialize invariant set
        self.update_invariant_set(verbose=True)

    def create_limit_lines(self, spacing=0.1):
        '''
        Creates the 4 boundary lines used in fast_equilibria()
        '''
        xmin, xmax, ymin, ymax = self.limits

        spam_x = np.arange(xmin, xmax, spacing)
        spam_y = np.arange(ymin, ymax, spacing)
        spam_x = spam_x.reshape( len(spam_x), 1 )
        spam_y = spam_y.reshape( len(spam_y), 1 )

        # Create 4 lines representing the window boundaries
        self.limit_lines = []
        
        top_x = spam_x
        top_y = ymax * np.ones(spam_x.shape)
        self.limit_lines.append({"x": top_x, "y": top_y})

        bottom_x = spam_x
        bottom_y = ymin * np.ones(spam_x.shape)
        self.limit_lines.append({"x": bottom_x, "y": bottom_y})

        left_x = xmin * np.ones(spam_y.shape)
        left_y = spam_y
        self.limit_lines.append({"x": left_x, "y": left_y})

        right_x = xmax * np.ones(spam_y.shape)
        right_y = spam_y
        self.limit_lines.append({"x": right_x, "y": right_y})

    def compute_cbf_boundary(self):
        '''Compute CBF boundary'''
        self.boundary_lines = []
        for boundary_seg in self.cbf.get_boundary():
            boundary_line = geometry.LineString(boundary_seg)
            self.boundary_lines.append(boundary_line)

    def set_param(self, **kwargs):
        '''
        Sets the following parameters: limits, spacing (for invariant set and equilibria computation)
                                       invariant_color, equilibria_color
                                       plant, clf, cbf, params
        '''
        for key in kwargs.keys():
            if key == "limits":
                self.limits = kwargs["limits"]
                continue
            if key == "spacing":
                self.spacing = kwargs["spacing"]
                continue
            if key == "invariant_color":
                self.invariant_color = kwargs["invariant_color"]
                continue
            if key == "equilibria_color":
                self.equilibria_color = kwargs["equilibria_color"]
                continue
            if key == "plant":
                self.plant = kwargs["plant"]
                continue
            if key == "clf":
                self.clf = kwargs["clf"]
                continue
            if key == "cbf":
                self.cbf = kwargs["cbf"]
                continue
            if key == "params":
                params = kwargs["params"]
                for key in params.keys():
                    if "slack_gain":
                        self.params["slack_gain"] = params["slack_gain"]
                    if "clf_gain":
                        self.params["clf_gain"] = params["clf_gain"]
                    if "cbf_gain":
                        self.params["cbf_gain"] = params["cbf_gain"]

        if "limits" not in kwargs.keys():
            if len(self.cbf.points) > 0:
                pts = np.array([ pt["coords"] for pt in self.cbf.points ])
                bbox = minimum_bounding_rectangle(pts)                
                xmin, xmax, ymin, ymax = np.min(bbox[:,0]), np.max( bbox[:,0] ), np.min(bbox[:,1]), np.max(bbox[:,1])
                self.limits = ( xmin, xmax, ymin, ymax )

        # Initialize grids used for determinant computation
        if hasattr(self, "limits") and hasattr(self, "spacing"):
            xmin, xmax, ymin, ymax = self.limits
            x = np.arange(xmin, xmax, self.spacing)
            y = np.arange(ymin, ymax, self.spacing)

            self.xg, self.yg = np.meshgrid(x,y)
            self.grid_shape = self.xg.shape
            self.grid_pts = list( zip( self.xg.flatten(), self.yg.flatten() ) )
            self.determinant_grid = np.empty(self.grid_shape, dtype=float)

        N = 10
        x = np.random.uniform(xmin, xmax, N)
        y = np.random.uniform(ymin, ymax, N)
        self.pts = np.column_stack((x, y))

        self.verify_kernel()
        self.counter = 0

    def verify_kernel(self):
        '''
        Verifies if the kernel pair is consistent and fully defined
        '''
        try:
            if not isinstance(self.plant, KernelAffineSystem) or not isinstance(self.plant.kernel, Kernel):
                raise Exception("Plant is not kernel affine.")
            if not isinstance(self.clf, KernelLyapunov) or not isinstance(self.clf.kernel, Kernel):
                raise Exception("CLF is not kernel-based.")
            if not isinstance(self.cbf, KernelBarrier) or not isinstance(self.cbf.kernel, Kernel):
                raise Exception("CBF is not kernel-based.")
            if not (self.plant.kernel == self.clf.kernel and self.plant.kernel == self.cbf.kernel):
                raise Exception("Kernels are not compatible.")

            self.kernel = self.plant.kernel

            if (self.kernel._dim != self.plant.n) or (self.kernel._dim != self.clf._dim) or (self.kernel._dim != self.cbf._dim):
                raise Exception("Dimensions are not compatible.")
            
            self.n = self.kernel._dim
            self.p = self.kernel._num_monomials
            self.A_matrices = self.kernel.get_A_matrices()

            self.F = self.plant.get_F()
            self.P = self.clf.P
            self.Q = self.cbf.Q
            self.ATQ_matrices = [ A.T @ self.Q for A in self.A_matrices ]
            
        except Exception as error:
            print(error)
            return False

    def vecQ_fun(self, pt: np.ndarray) -> np.ndarray:
        '''Returns the vector vQ = ∇h'''    
        m = self.kernel.function(pt)
        return np.array([ m.T @ ATQ @ m for ATQ in self.ATQ_matrices ])

    def vecP_fun_with_shape(self, pt: np.ndarray, P) -> list:
        '''Returns the vector vP = p gamma V(x, self.P) ∇V - fc '''

        m = self.kernel.function(pt)
        V = self.clf_fun(pt)
        slk_gain = self.params["slack_gain"]
        clf_gain = self.params["clf_gain"]

        vecP = [ m.T @ A.T @ ( slk_gain * clf_gain * V * P - self.F ) @ m for A in self.A_matrices ]
        return vecP
    
    def vecP_fun(self, pt: np.ndarray) -> np.ndarray:
        '''Returns the vector vP = p gamma V(x, self.P) ∇V - fc with self P matrix'''
        return np.array(self.vecP_fun_with_shape(pt, self.P))

    def clf_fun_with_shape(self, pt: np.ndarray, P):
        '''Returns the value of the CLF with shape defined by matrix P'''
        m = self.kernel.function(pt)
        return 0.5 * m.T @ P @ m

    def clf_gradient_with_shape(self, pt: np.ndarray, P):
        '''Returns the gradient of the CLF with shape defined by matrix P'''
        m = self.kernel.function(pt)
        Jm = self.kernel.jacobian(pt)
        return Jm.T @ P @ m

    def clf_fun(self, pt: np.ndarray) -> float:
        '''Returns the CLF value using self P matrix'''
        return self.clf_fun_with_shape(pt, self.P)

    def clf_gradient(self, pt: np.ndarray) -> np.ndarray:
        '''Returns the CLF gradient using self P matrix'''  
        return self.clf_gradient_with_shape(pt, self.P)

    def lambda_fun(self, pt):
        '''Given a point x in the invariant set, compute its corresponding lambda scalar.'''
        vQ = self.vecQ_fun(pt)
        vP = self.vecP_fun(pt)
        return (vQ.T @ vP) / np.linalg.norm(vQ)**2

    def L_fun_with_lambda_and_shape(self, pt: np.ndarray, l, P):
        '''Returns matrix L = F + l Q - p gamma V(x, self.P) P'''

        slk_gain = self.params["slack_gain"]
        clf_gain = self.params["clf_gain"]
        V = self.clf_fun(pt)

        return self.F + l * self.Q - slk_gain * clf_gain * V * P

    def L_fun(self, pt):
        '''Returns matrix L = F + l(x) Q - p gamma V(x, self.P) P with l(pt) and self P matrix'''
        return self.L_fun_with_lambda_and_shape( pt, self.lambda_fun(pt), self.P )

    def invariant_equation(self, pt: np.ndarray, l, P):
        '''Returns invariant equation l vQ - vP for a given pt, l and P'''
        m = self.kernel.function(pt)
        Jm = self.kernel.jacobian(pt)
        return Jm.T @ self.L_fun_with_lambda_and_shape(pt, l, P) @ m

    def S_fun_with_lambda_and_shape(self, pt: np.ndarray, l, P):
        '''Returns matrix S = H - (1/p gamma V**2) * fc fc.T, for stability computation of equilibrium points'''

        V = self.clf_fun(pt)
        m = self.kernel.function(pt)
        fc = self.plant.get_fc(pt)

        L = self.L_fun_with_lambda_and_shape(pt, l, P)

        slk_gain = self.params["slack_gain"]
        clf_gain = self.params["clf_gain"]

        S = []
        for i, Ai in enumerate(self.A_matrices):
            S.append([])
            for j, Aj in enumerate(self.A_matrices):
                S[-1].append( m.T @ Ai.T @ ( L @ Aj + Aj.T @ L ) @ m - fc[i]*fc[j] / ( slk_gain * clf_gain * ( V**2 ) ) )
        return S

    def S_fun(self, pt: np.ndarray) -> np.ndarray:
        '''Returns matrix S = H - (1/p gamma V**2) * fc fc.T with l(pt) and self P matrix'''
        return np.array(self.S_fun_with_lambda_and_shape(pt, self.lambda_fun(pt), self.P))

    def stability_fun(self, x_eq, type_eq): 
        '''Compute the stability number for a given equilibrium point'''
        V = self.clf_fun(x_eq)
        nablaV = self.clf_gradient(x_eq)
        nablah = self.cbf.gradient(x_eq)
        norm_nablaV = np.linalg.norm(nablaV)
        norm_nablah = np.linalg.norm(nablah)
        unit_nablah = nablah/norm_nablah

        S = self.S_fun(x_eq)
        if type_eq == "boundary":
            curvatures, basis_for_TpS = compute_curvatures( S, unit_nablah )
            max_index = np.argmax(curvatures)
            stability_number = curvatures[max_index] / ( self.params["slack_gain"] * self.params["clf_gain"] * V * norm_nablaV )

        if type_eq == "interior":
            stability_number = np.max( np.linalg.eigvals(S) )

        '''
        If the CLF-CBF gradients are collinear, then the stability_number is equivalent to the diff. btw CBF and CLF curvatures at the equilibrium point:
        '''
        eta = self.eta_fun(x_eq)
        # if (eta - 1) < 1e-10:
        #     curv_V = self.clf.get_curvature(x)
        #     curv_h = self.cbf.get_curvature(x)
        #     diff_curvatures = curv_h - curv_V
        #     print(f"Difference of curvatures = {diff_curvatures}")
        #     print(f"Stability = {stability_number}")
        #     if np.abs(diff_curvatures - stability_number) > 1e-3:
        #         raise Exception("Stability number is different then the difference of curvatures.")

        return stability_number, eta

    def eta_fun(self, x_eq):
        '''
        Returns the value of eta (between 0 and 1), depending on the collinearity between the CLF-CBF gradients.
        '''
        nablaV = self.clf_gradient(x_eq)
        nablah = self.cbf.gradient(x_eq)

        g = self.plant.get_g(x_eq)
        G = g @ g.T
        z1 = nablah / np.linalg.norm(nablah)
        z2 = nablaV - nablaV.T @ G @ z1 * z1
        eta = 1/(1 + self.params["slack_gain"] * z2.T @ G @ z2 )
        return eta

    def local_minimize_cost(self, cost, init_pt: np.ndarray) -> np.ndarray:
        '''Locally minimizes a passed cost function, starting with initial guess init_pt'''
        sol = minimize( cost, init_pt )
        return sol.x

    def find_closest_valid(self, Pnom: np.ndarray, verbose=False) -> np.ndarray:
        ''' Find closest shape matrix to Pnom belongin to family of valid CLFs (without local minima) '''

        self.CVXPY_Pnom.value = Pnom

        if np.any( np.linalg.eigvals(self.kernel.get_left_lowerbound(Pnom)) != 0 ):
            self.CVXPY_family_problem.solve(verbose=verbose)
            return self.CVXPY_P.value
        else:
            return self.CVXPY_Pnom.value

    def update_determinant_grid(self):
        '''
        Evaluates det([ vQ, vP ]) = 0 over a grid.
        Returns: a grid of shape self.grid_shape with the determinant values corresponding to each pt on the grid
        '''
        self.W_list, self.lambda_grid = [], []
        for pt in self.grid_pts:
            vQ = self.vecQ_fun(pt)
            vP = self.vecP_fun(pt)

            W = np.hstack([vQ.reshape(self.n,1), vP.reshape(self.n,1)])
            self.W_list.append(W)

            self.lambda_grid.append( (vQ.T @ vP) / np.linalg.norm(vQ)**2 )

        determinant_list = np.linalg.det( self.W_list )

        # Eliminate the negative lambda part
        if not self.invariant_complete:
            for k, l in enumerate(self.lambda_grid):
                if l < 0.0: determinant_list[k] = np.inf

        self.determinant_grid = determinant_list.reshape(self.grid_shape)

    def update_invariant_set(self, verbose=False):
        '''
        Computes the invariant set for the given CLF-CBF pair.
        '''
        if self.n > 2:
            warnings.warn("Currently, the computation of the invariant set is not available for dimensions higher than 2.")
            return
        
        self.update_determinant_grid()                                                                            # updates the grid with new determinant values
        invariant_contour = ctp.contour_generator( x=self.xg, y=self.yg, z=self.determinant_grid )  # creates new contour_generator object
        self.invariant_lines = invariant_contour.lines(0.0)                                                           # returns the 0-valued contour lines
        self.invariant_set_analysis(verbose=verbose)                                                                  # run through each branch of the invariant set

    def invariant_set_analysis(self, verbose=False):
        '''
        Populates invariant segments with data and compute equilibrium points from invariant line data.
        '''
        self.invariant_segs = []
        self.boundary_equilibria = []
        self.interior_equilibria = []
        self.stable_equilibria = []
        self.unstable_equilibria = []

        for segment_points in self.invariant_lines:

            # ----- Loads segment dictionary
            seg_dict = {"points": segment_points}
            seg_dict["lambdas"] = [ self.lambda_fun(pt) for pt in segment_points ]

            seg_dict["clf_values"] = [ self.clf_fun(pt) for pt in segment_points ]
            seg_dict["clf_gradients"] = [ self.clf_gradient(pt) for pt in segment_points ]

            seg_dict["cbf_values"] = [ self.cbf.function(pt) for pt in segment_points ]
            seg_dict["cbf_gradients"] = [ self.cbf.gradient(pt) for pt in segment_points ]

            # ----- Computes the corresponding equilibrium points and critical segment values
            self.seg_boundary_equilibria(seg_dict)
            self.seg_interior_equilibria(seg_dict)
            self.seg_critical(seg_dict)

            # ----- Adds the segment dicts and equilibrium points to corresponding data structures
            self.invariant_segs.append(seg_dict)
            self.boundary_equilibria += seg_dict["boundary_equilibria"]
            self.interior_equilibria += seg_dict["interior_equilibria"]

        for b_eq in self.boundary_equilibria:
            if b_eq["equilibrium"] == "stable": self.stable_equilibria.append(b_eq)
            if b_eq["equilibrium"] == "unstable": self.unstable_equilibria.append(b_eq)

        for i_eq in self.interior_equilibria:
            if i_eq["equilibrium"] == "stable": self.stable_equilibria.append(i_eq)
            if i_eq["equilibrium"] == "unstable": self.unstable_equilibria.append(i_eq)

        if verbose:
            show_message(self.boundary_equilibria, "boundary equilibrium points")
            show_message(self.interior_equilibria, "interior equilibrium points")

    def get_boundary_intersections(self, seg_data: list[np.ndarray]):
        ''' Computes the intersections with boundary segments of a particular segment of the invariant set '''

        intersection_pts = []
        for boundary_line in self.boundary_lines:
            invariant_seg_line = geometry.LineString(seg_data)
            intersections = intersection( boundary_line, invariant_seg_line )

            new_candidates = []
            if not intersections.is_empty:
                if hasattr(intersections, "geoms"):
                    for geo in intersections.geoms:
                        x, y = geo.xy
                        x, y = list(x), list(y)
                        new_candidates += [ [x[k], y[k]] for k in range(len(x)) ]
                else:
                    x, y = intersections.xy
                    x, y = list(x), list(y)
                    new_candidates += [ [x[k], y[k]] for k in range(len(x)) ]

            intersection_pts += new_candidates
        
        return intersection_pts

    def seg_boundary_equilibria(self, seg_dict: dict[str,]):
        ''' Computes boundary equilibrium points for given segment data '''

        seg_dict["boundary_equilibria"] = []
        intersection_pts = self.get_boundary_intersections(seg_dict["points"])

        # Boundary equilibrium are intersection points with positive lambda:
        for pt in intersection_pts:
            lambda_pt = self.lambda_fun(pt)
            if lambda_pt >= 0.0:
                seg_boundary_equilibrium = {"x": pt}
                seg_boundary_equilibrium["lambda"] = lambda_pt
                seg_boundary_equilibrium["h"] = self.cbf.function(pt)
                seg_boundary_equilibrium["nablah"] = self.cbf.gradient(pt)

                stability, eta = self.stability_fun(pt, "boundary")
                seg_boundary_equilibrium["eta"], seg_boundary_equilibrium["stability"] = eta, stability
                seg_boundary_equilibrium["equilibrium"] = "stable"
                if stability > 0: seg_boundary_equilibrium["equilibrium"] = "unstable"

                seg_dict["boundary_equilibria"].append( seg_boundary_equilibrium )

    def seg_interior_equilibria(self, seg_dict: dict[str,]):
        ''' Computes interior equilibrium points for given segment data '''

        seg_dict["interior_equilibria"] = []
        seg_data = seg_dict["points"]

        slk_gain = self.params["slack_gain"]
        clf_gain = self.params["clf_gain"]

        # Computes the costs along the whole segment
        costs = []
        for k, (V, nablaV) in enumerate(zip(seg_dict["clf_values"], seg_dict["clf_gradients"])):
            pt = seg_data[k]
            fc = self.plant.get_fc(pt)
            costs.append( np.linalg.norm( fc - slk_gain * clf_gain * V * nablaV ) )

        # Finds separate groups of points with costs below a certain threshold... interior equilibria are computed by extracting the argmin of the cost for each group
        for flag, group in itertools.groupby(zip(seg_data, costs), lambda x: x[1] <= self.interior_eq_threshold):
            if flag:
                group = list(group)
                group_pts = [ ele[0] for ele in group ]
                group_costs = [ ele[1] for ele in group ]
                new_eq = group_pts[np.argmin(group_costs)]
                seg_dict["interior_equilibria"].append({"x":new_eq, 
                                                        "lambda": self.lambda_fun(new_eq), 
                                                        "h": self.cbf.function(new_eq), 
                                                        "nablah": self.cbf.gradient(new_eq)})

        # Computes the equilibrium stability
        for eq in seg_dict["interior_equilibria"]:
            stability, eta = self.stability_fun(eq["x"], "boundary")
            eq["eta"], eq["stability"] = eta, stability
            eq["equilibrium"] = "stable"
            if stability > 0:
                eq["equilibrium"] = "unstable"
        
    def seg_clf_minima(self, seg_data: list[np.ndarray]):
        '''
        Computes CLF local minima for given segment data
        '''

    def seg_cbf_minima(self, seg_data: list[np.ndarray]):
        '''
        Computes CBF local minima for given segment data
        '''

    def seg_critical(self, seg_dict: dict[str,]) -> float:
        '''
        Computes the segment integral
        '''
        seg_data = seg_dict["points"]
        barrier_vals = np.array([ self.cbf.function(0.5*( pt + seg_data[k+1,:] )) for k, pt in enumerate(seg_data[0:-1,:]) ])

        # segment is removable (starts AND ends completely outside/inside the unsafe set)
        if barrier_vals[0] * barrier_vals[-1] > 0:

            if barrier_vals[0] > 0:                                                 # removable from outside
                seg_dict["removable"] = +1
                seg_dict["segment_critical"] = np.min(barrier_vals)
                return
            
            if barrier_vals[0] < 0:                                                 # removable from inside
                seg_dict["removable"] = -1
                seg_dict["segment_critical"] = np.max(barrier_vals)
                return

        # segment is not removable (starts OR ends outside/inside the unsafe set)
        seg_dict["removable"] = 0
        pos_lines = np.where(barrier_vals >= self.compatibility_options["barrier_sep"])
        if len(pos_lines[0]) == 0: pos_lines = np.where(barrier_vals >= 0.0)

        if barrier_vals[0] < 0:                                                                     # starts inside (ends outside)
            sep_index = pos_lines[0][0]
            removable_line = barrier_vals[sep_index:]

        if barrier_vals[0] > 0:                                                                     # starts outside (ends inside)
            sep_index = pos_lines[0][-1]+1
            removable_line = barrier_vals[0:sep_index]

        seg_dict["segment_critical"] = np.min(removable_line)
        return 

    def is_compatible(self) -> bool:
        '''
        Checks if kernel triplet is compatible.
        '''
        # Checks if boundary has stable equilibria
        for boundary_eq in self.boundary_equilibria:
            if boundary_eq["equilibrium"] == "stable":
                return False
            
        # Checks if more than one stable interior equilibria exist 
        stable_interior_counter = 0
        for interior_eq in self.interior_equilibria:
            if interior_eq["equilibrium"] == "stable": stable_interior_counter += 1
        if stable_interior_counter > 1: return False

        return True

    def fit_curvatures(self, points: list[np.ndarray], Pinit: np.ndarray) -> np.ndarray:
        '''
        Solves the convex optimization problem of finding the closest P matrix to Pinit s.t. 
        all input points are with unstable boundary equilibria.

        Parameters: points - boundary equilibrium points
                    Pinit - initial P matrix

        Returns: P - the final result of optimization. 
        '''
        if len(points) == 0: return self.P

        self.CVXPY_Pnom.value = Pinit
        for pt in points:

            # Level set constraint
            self.CVXPY_constraints.append( self.clf_fun_with_shape(pt, self.CVXPY_P) == self.clf_fun_with_shape(pt, Pinit) )

            # Equilibrium point constraint (the equilibrium point locations must be constant)
            self.CVXPY_lambdas.append( cp.Variable(pos=True) )
            CVXPY_lambda = self.CVXPY_lambdas[-1]
            self.CVXPY_constraints.append( self.invariant_equation(pt, CVXPY_lambda, self.CVXPY_P) == 0 )

            # Curvature constraint for equilibrium point instabilization
            init_gradient = self.clf_gradient_with_shape(pt, Pinit)
            norm_init_gradient = np.linalg.norm(init_gradient)
            v = rot2D(np.pi/2) @ init_gradient/norm_init_gradient

            S = self.S_fun_with_lambda_and_shape(pt, CVXPY_lambda, self.CVXPY_P)
            self.CVXPY_constraints.append( v.T @ S @ v / norm_init_gradient - self.compatibility_options["min_curvature"] >= 0 )

        fit_problem = cp.Problem( cp.Minimize( self.CVXPY_cost ), self.CVXPY_constraints )
        fit_problem.solve(verbose=False)

        return self.CVXPY_P.value

    def non_removable_stable_equilibria(self) -> list[np.ndarray]:
        '''
        This function returns a list with all non-removable stable equilibrium points.
        '''
        nonremovable_stables = []
        for seg_dict in self.invariant_segs:
            if seg_dict["removable"] == 0:
                for eq in seg_dict["boundary_equilibria"]:
                    if eq["equilibrium"] == "stable":
                        nonremovable_stables.append( np.array(eq["x"]) )
        return nonremovable_stables

    def compatibilize(self, obj_type="closest", verbose=False, animate=False) -> dict:
        '''
        This function computes a new CLF geometry that is completely compatible with the original CBF.
        '''
        np.set_printoptions(precision=4, suppress=True)

        is_original_compatible = self.is_compatible()
        self.P = self.clf.P
        Pnom = self.clf.P
        self.counter = 0

        def var_to_PSD(var: np.ndarray) -> np.ndarray:
            '''Transforms an n(n+1)/2 array representing a stacked symmetric matrix into standard PSD form'''
            sqrtP = vector2sym(var)
            P = sqrtP.T @ sqrtP
            return P

        def PSD_to_var(P: np.ndarray) -> np.ndarray:
            '''Transforms a standard PSD matrix P into an array of size n(n+1)/2 list representing the stacked symmetric square root matrix of P'''
            return sym2vector(sp.linalg.sqrtm(P))

        def objective(var: np.ndarray) -> float:
            ''' Minimizes the changes to the CLF geometry needed for compatibilization '''
            self.counter += 1
            P = var_to_PSD(var)
            if obj_type == "closest": self.cost = np.linalg.norm( P - Pnom, 'fro')
            else: self.cost = 1.0

            return self.cost

        def lowerbound_constr(var: np.ndarray) -> list[float]:
            ''' Avoids creation of undesirable interior equilibria '''
            P = var_to_PSD(var)
            self.lowerb_constr = [ np.linalg.norm(P @ col) for col in self.kernel.Asum2.T if np.any(col != 0.0) ]
            return self.lowerb_constr

        def lambda_max_constr(var: np.ndarray) -> list[float]:
            ''' Avoids eigenvalues of P from exploding '''
            P = var_to_PSD(var)
            return self.max_shape_eigenvalue - max(np.linalg.eigvals(P))

        def removability_constr(var: np.ndarray) -> list[float]:
            ''' Removes removable branches '''
            self.P = var_to_PSD(var)
            
            self.rem_constr = [ 0.0, 0.0 ]
            min_barrier_values, max_barrier_values = [], []

            # The invariant set must be updated to get the 
            self.update_invariant_set()
            for seg in self.invariant_segs:
                if seg["removable"] >= 0 :
                    min_barrier_values.append( seg["segment_critical"] )        # if segment is not removable or removable from outside
                if seg["removable"] == -1: 
                    max_barrier_values.append( seg["segment_critical"] )        # if segment is removable from inside

            if len(min_barrier_values): self.rem_constr[0] =  np.min( min_barrier_values ) - self.compatibility_options["barrier_sep"]
            if len(max_barrier_values): self.rem_constr[1] = -np.max( max_barrier_values ) - self.compatibility_options["barrier_sep"]

            return self.rem_constr

        if animate:
            self.comp_graphics["fig"], ax = plt.subplots(nrows=1, ncols=1)

            ax.set_title("Showing compatibilization process...")
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(self.limits[0], self.limits[1])
            ax.set_ylim(self.limits[2], self.limits[3])
            self.init_comp_plot(ax)
            plt.pause(self.comp_process_data["gui_eventloop_time"])
        
        def intermediate_callback(res: np.ndarray):
            '''
            Callback for visualization of intermediate results (verbose or by animation).
            '''
            # print(f"Status = {status}")
            self.comp_process_data["execution_time"] += time.perf_counter() - self.comp_process_data["start_time"]
            self.comp_process_data["step"] += 1
            
            # for later json serialization...
            self.comp_process_data["invariant_segs_log"].append( [ seg.tolist() for seg in self.invariant_lines ] )

            if verbose:
                print( f"Steps = {self.counter}" )
                print( f"λ(P) = {np.linalg.eigvals(self.P)}" )

                lambdaM = self.kernel.get_lowerbound(self.P)
                print( f"λ(M(P)) = {np.linalg.eigvals(lambdaM)}" )

                print( f"Cost = {self.cost}" )
                print( f"Lowerbound constraints = {self.lowerb_constr}" )
                print( f"Removability constraints = {self.rem_constr}" )
                print(self.comp_process_data["execution_time"], "seconds have passed...")

            if animate: 
                self.update_comp_plot(ax)
                plt.pause(self.comp_process_data["gui_eventloop_time"])

            self.counter = 0
            self.comp_process_data["start_time"] = time.perf_counter()

        constraints = [ {"type": "eq", "fun": lowerbound_constr} ]
        constraints += [ {"type": "ineq", "fun": lambda_max_constr} ]
        constraints += [ {"type": "ineq", "fun": removability_constr} ]

        #--------------------------- Main compatibilization process ---------------------------
        print("Starting compatibilization process. This may take a while...")
        is_processed_compatible = self.is_compatible()
        self.comp_process_data["start_time"] = time.perf_counter()

        # while not is_processed_compatible:

        init_var = PSD_to_var(self.P)
        sol = minimize( objective, init_var, constraints=constraints, callback=intermediate_callback, options={"ftol": 1e-4} )

        self.P = var_to_PSD( sol.x )
        self.update_invariant_set()

        is_processed_compatible = self.is_compatible()
        # --------------------------- Main compatibilization process ---------------------------
        print(f"Compatibilization terminated with message: {sol.message}")

        message = "Compatibilization "
        if is_processed_compatible: message += "was successful. "
        else: message += "failed. "
        message += "Process took " + str(self.comp_process_data["execution_time"]) + " seconds."
        print(message)

        if animate: plt.pause(2)

        comp_result = { 
                        # "opt_message": sol.message, 
                        "kernel_dimension": self.kernel._num_monomials,
                        "P_original": Pnom.tolist(),
                        "P_processed": self.P.tolist(),
                        "is_original_compatible": is_original_compatible,
                        "is_processed_compatible": is_processed_compatible,
                        "execution_time": self.comp_process_data["execution_time"],
                        "num_steps": self.comp_process_data["step"],
                        "invariant_set_log": self.comp_process_data["invariant_segs_log"] 
                    }
    
        return comp_result

    def plot_invariant(self, ax, *args):
        '''
        Plots the invariant set segments into ax.
        Optional arguments specify the indexes of each invariant segment to be plotted.
        If no optional argument is passed, plots all invariant segments.
        '''
        # Which segments to plot?
        num_segs_to_plot = len(self.invariant_segs)
        segs_to_plot = [ i for i in range(num_segs_to_plot) ]

        if np.any( np.array(args) > len(self.invariant_segs)-1 ):
            print("Invariant segment list index out of range. Plotting all")

        elif len(args) > 0:
            num_segs_to_plot = len(args)
            segs_to_plot = list(args)

        # Adds or removes lines according to the total number of segments to be plotted 
        if num_segs_to_plot >= len(self.invariant_lines_plot):
            for _ in range(num_segs_to_plot - len(self.invariant_lines_plot)):
                line2D, = ax.plot([],[], color=np.random.rand(3), linestyle='dashed', linewidth=1.2 )
                self.invariant_lines_plot.append(line2D)
        else:
            for _ in range(len(self.invariant_lines_plot) - num_segs_to_plot):
                self.invariant_lines_plot[-1].remove()
                del self.invariant_lines_plot[-1]

        # UP TO HERE: len(self.invariant_lines_plot) == len(segs_to_plot)

        # Updates segment lines with data from each invariant segment
        for k in range(num_segs_to_plot):
            seg_index = segs_to_plot[k]
            self.invariant_lines_plot[k].set_data( self.invariant_segs[seg_index]["points"][:,0], self.invariant_segs[seg_index]["points"][:,1] )

    def plot_attr(self, ax, attr_name: str, plot_color='k', alpha=1.0):
        '''
        Plots a list attribute from the class into ax.
        '''
        attr = getattr(self, attr_name)
        if type(attr) != list:
            raise Exception("Passed attribute name is not a list.")
        
        # If the passed attribute name was not already initialized, initialize it.
        if attr_name not in self.plotted_attrs.keys():
            self.plotted_attrs[attr_name] = []

        # Balance the number of line2D elements in the array of plotted points
        if len(attr) >= len(self.plotted_attrs[attr_name]):
            for _ in range(len(attr) - len(self.plotted_attrs[attr_name])):
                line2D, = ax.plot([],[], 'o', color=plot_color, alpha=alpha, linewidth=0.6 )
                self.plotted_attrs[attr_name].append(line2D)
        else:
            for _ in range(len(self.plotted_attrs[attr_name]) - len(attr)):
                self.plotted_attrs[attr_name][-1].remove()
                del self.plotted_attrs[attr_name][-1]

        # from this point on, len(attr) = len(self.plotted_attrs[attr_name])
        for k in range(len(attr)):
            self.plotted_attrs[attr_name][k].set_data( attr[k]["x"][0], attr[k]["x"][1] )

    def init_comp_plot(self, ax):
        ''' Initialize compatibilization animation plot '''
        self.comp_graphics["text"] = ax.text(0.01, 0.99, str("Optimization step = 0"), ha='left', va='top', transform=ax.transAxes, fontsize=10)
        self.cbf.plot_levels(levels = [ -0.1*k for k in range(4,-1,-1) ], ax=ax, limits=self.limits)
        self.update_comp_plot(ax)

    def update_comp_plot(self, ax):
        ''' Update compatibilization animation plot '''
        
        step = self.comp_process_data["step"]
        self.comp_graphics["text"].set_text(f"Optimization step = {step}")

        self.plot_invariant(ax)
        self.plot_attr(ax, "stable_equilibria", mcolors.BASE_COLORS["r"], 1.0)
        self.plot_attr(ax, "unstable_equilibria", mcolors.BASE_COLORS["g"], 0.8)

        for coll in self.comp_graphics["clf_artists"]:
            coll.remove()

        num_eqs = len(self.boundary_equilibria)
        if num_eqs:
            self.clf.set_params(P=self.P)
            self.clf.generate_contour()
            level = self.clf.function( self.boundary_equilibria[np.random.randint(0,num_eqs)]["x"] )
            self.comp_graphics["clf_artists"] = self.clf.plot_levels(levels = [ level ], ax=ax, limits=self.limits)

class CLBF(KernelQuadratic):
    '''
    Class for kernel-based Control Lyapunov Barrier Functions.
    '''
    def __init__(self, *args):
        super().__init__(*args)
        pass

##################### OLD CODE #############################################################################################################

# class Function():
#     '''
#     Implementation of general class for scalar functions.
#     '''
#     def __init__(self, *args, **kwargs):

#         self.set_value(*args)
#         self.functions = []
#         self.functions.append(self)

#         self._function = 0.0
#         if self._dim > 1:
#             self._gradient = np.zeros(self._dim)
#             self._hessian = np.zeros([self._dim,self._dim])
#         else:
#             self._gradient = 0.0
#             self._hessian = 0.0

#         limits = None
#         spacing = 0.1
#         self.plot_config = {"color": mcolors.BASE_COLORS["k"], "linestyle": 'solid'}
#         for key in kwargs.keys():
#             if key == "limits":
#                 limits = kwargs["limits"]
#                 continue
#             if key == "spacing":
#                 spacing = kwargs["spacing"]
#                 continue
#             if key == "plot_config":
#                 self.plot_config = kwargs["plot_config"]
#                 continue

#         if limits != None:
#             self.gen_contour(limits, spacing=spacing)

#     def set_value(self, *args):
#         '''
#         Initialize values and returns corresponding points.
#         '''
#         data_type = np.array(args).dtype
#         if data_type != np.dtype('float64') and data_type != np.dtype('int64'):
#             raise Exception("Data type not understood.")

#         self._args = args
#         self._dim = len(args)
#         self._var = np.array(self._args).reshape(self._dim,-1)
#         self._num_points = np.size(self._var, 1)

#         return self.get_value()

#     def get_value(self):
#         return self._var

#     def evaluate(self):
#         self.evaluate_function(*self._var)
#         self.evaluate_gradient(*self._var)
#         self.evaluate_hessian(*self._var)

#     def evaluate_function(self, *args):
#         self.set_value(*args)
#         self.function_values()
#         return self.get_function()

#     def evaluate_gradient(self, *args):
#         self.set_value(*args)
#         self.gradient_values()
#         return self.get_gradient()

#     def evaluate_hessian(self, *args):
#         self.set_value(*args)
#         self.hessian_values()
#         return self.get_hessian()

#     def get_function(self):
#         '''
#         Get last computed function
#         '''
#         return self._function

#     def get_gradient(self):
#         '''
#         Get last computed gradient
#         '''
#         return self._gradient

#     def get_hessian(self):
#         '''
#         Get last computed hessian
#         '''
#         return self._hessian

#     def function_values(self):
#         '''
#         Compute function values.
#         '''
#         self._function = np.zeros(self._num_points)
#         for k in range(self._num_points):
#             fun_val = 0.0
#             for func in self.functions:
#                 fun_val += func.function(self._var[:,k])
#             self._function[k] = fun_val

#         return self._function

#     def gradient_values(self):
#         '''
#         Compute gradient values.
#         '''
#         self._gradient = []
#         for point in self._var.T:
#             grad = np.zeros(self._dim)
#             for func in self.functions:
#                 grad += func.gradient(point)
#             self._gradient.append(grad)

#         return self._gradient

#     def hessian_values(self):
#         '''
#         Compute hessian values.
#         '''
#         self._hessian = []
#         for point in self._var.T:
#             hess = np.zeros([self._dim,self._dim])
#             for func in self.functions:
#                 hess += func.gradient(point)
#             self._hessian.append(hess)

#         return self._hessian

#     def function(self, point):
#         '''
#         Abstract implementation of function computation. Must receive point as input and return the corresponding function value.
#         Overwrite on children classes.
#         '''
#         return 0.0

#     def gradient(self, point):
#         '''
#         Abstract implementation of gradient computation. Must receive point as input and return the corresponding gradient value.
#         Overwrite on children classes.
#         '''
#         return np.zeros(self._dim)

#     def hessian(self, point):
#         '''
#         Abstract implementation of hessian computation. Must receive point as input and return the corresponding hessian value.
#         Overwrite on children classes.
#         '''
#         return np.zeros([self._dim, self._dim])

#     def __add__(self, func):
#         '''
#         Add method.
#         '''
#         if not isinstance(func, Function):
#             raise Exception("Only Function objects can be summed.")

#         from copy import copy
#         function = copy(self)
#         function.functions.append(func)

#         return function

#     def gen_contour(self, limits, spacing=0.1):
#         '''
#         Create contour generator object for the given function.
#         Parameters: limits (2x2 array) - min/max limits for x,y coords
#                     spacing - grid spacing for contour generation
#         '''        
#         if self._dim != 2:
#             raise Exception("Contour plot can only be used for 2D functions.")

#         x_min, x_max = limits[0][0], limits[0][1]
#         y_min, y_max = limits[1][0], limits[1][1]

#         x = np.arange(x_min, x_max, spacing)
#         y = np.arange(y_min, y_max, spacing)
#         xg, yg = np.meshgrid(x,y)

#         mesh_fvalues = np.zeros([np.size(xg,0),np.size(xg,1)])
#         for i in range(np.size(xg,1)):
#             args = []
#             args.append(xg[:,i])
#             args.append(yg[:,i])
#             for k in range(self._dim-2):
#                 args.append( [self._var[k+2,0] for _ in range(len(xg[:,i]))] )
#             # mesh_fvalues[:,i] = np.array(self.evaluate_function(xv[:,i], yv[:,i]))
#             mesh_fvalues[:,i] = np.array(self.evaluate_function(*args))
        
#         self.contour = ctp.contour_generator(x=xg, y=yg, z=mesh_fvalues )
#         return self.contour

#     def get_levels(self, levels, **kwargs):
#         '''
#         Generates function level sets.
#         Parameters: levels (list of floats)
#         Returns: a list with all level segments, in the same order as levels
#         '''
#         limits = None
#         spacing = 0.1
#         for key in kwargs.keys():
#             aux_key = key.lower()
#             if aux_key == "limits":     # Must always be required if self.contours still does not exist
#                 limits = kwargs[key]
#                 continue
#             if aux_key == "spacing":    # Must always be required if self.contours still does not exist
#                 spacing = kwargs[key]
#                 continue

#         if not isinstance(limits, list):
#             if not hasattr(self, "contour"):
#                 raise Exception("Grid limits are required to create contours.")
#         else:
#             self.gen_contour(limits, spacing=spacing)

#         level_contours = []
#         for lvl in levels:
#             level_contours.append( self.contour.lines(lvl) )

#         return level_contours

#     def plot_levels(self, levels, **kwargs):
#         '''
#         Plots function level sets.
#         Parameters: levels (list of floats)
#         Returns: plot collections
#         '''
#         ax = plt
#         color = self.plot_config["color"]
#         linestyle = self.plot_config["linestyle"]
#         for key in kwargs.keys():
#             aux_key = key.lower()
#             if aux_key == "ax":
#                 ax = kwargs["ax"]
#                 continue
#             if aux_key == "color":
#                 color = kwargs["color"]
#                 continue
#             if aux_key == "linestyle":
#                 linestyle = kwargs["linestyle"]
#                 continue

#         collections = []
#         level_contours = self.get_levels(levels, **kwargs)
#         for level in level_contours:
#             for segment in level:
#                 line2D = ax.plot( segment[:,0], segment[:,1], color=color, linestyle=linestyle )
#                 collections.append(line2D[0])

#         return collections


# old methods from KernelTriplet
# class OLDKernelTripletMethops(KernelTriplet):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def update_segment(self, segment):
#         '''
#         Updates segment. Aims to significantly improve performance on updating the invariant set. 
#         '''
#         n = self.plant.n
#         seg_points = segment["points"]

#         num_pts = len(seg_points)
#         line_sizes = [ np.linalg.norm(seg_points[k] - seg_points[k+1]) for k in range(len(seg_points)-1) ]

#         def get_deltas(var):
#             return var[0:n*num_pts].reshape(num_pts, n)

#         def objective(var):
#             '''
#             var is a list with coordinates [ pt1[0] pt1[1] .. pt1[n-1] pt2[0] pt2[1] ... pt2[n-1] ...  ptm[n-1] l1 l2 ... lm ]
#             n is the state dimension
#             m is the number of points in the segment
#             '''
#             deltas = get_deltas(var)
#             fun = sum( map(lambda d: np.linalg.norm(d)**2, deltas) )

#             new_seg_points = seg_points + deltas
#             fun += sum( [ ( np.linalg.norm(new_seg_points[k] - new_seg_points[k+1]) - line_sizes[k] )**2 for k in range(num_pts-1) ] )
#             fun += sum( [ self.det_invariant( *new_seg_points[k].tolist() )**2 for k in range(num_pts) ] )
#             return fun
        
#         # def invariant_constr(var):
#         #     deltas = get_deltas(var)
#         #     return [ self.det_invariant( *(seg_points[k] + deltas[k]).tolist() ) for k in range(num_pts) ]
    
#         def lambda_constr(var):
#             deltas = get_deltas(var)
#             return [ self.compute_lambda( seg_points[k] + deltas[k] ) for k in range(num_pts) ]

#         # constraints = [ {"type": "eq", "fun": invariant_constr} ]
#         constraints = []
#         constraints.append( {"type": "ineq", "fun": lambda_constr} )

#         init_var = [ 0.0 for _ in range(n*num_pts) ]
#         sol = minimize(objective, init_var, constraints=constraints, options={"disp":True})

#         new_seg_pts = seg_points + get_deltas(sol.x)

#         segment["points"] = new_seg_pts
#         segment["lambdas"] = [ self.compute_lambda(new_seg_pts[k]) for k in range(num_pts) ]
#         segment["boundary_equilibria"] = self.seg_boundary_equilibria(new_seg_pts)
#         segment["interior_equilibria"] = self.seg_interior_equilibria(new_seg_pts)
#         return segment

#     def update_invariant_set(self):
#         '''
#         Updates the invariant set.
#         '''
#         self.boundary_equilibria = []
#         self.interior_equilibria = []
#         for segment in self.invariant_segs:
#             self.update_segment( segment )
#             self.boundary_equilibria += segment["boundary_equilibria"]
#             self.interior_equilibria += segment["interior_equilibria"]

#     def equilibria_from_invariant(self, verbose=False):
#         '''
#         Computes all equilibrium points and local branch optimizers of the CLF-CBF pair, using the invariant set intersections with the CBF boundary.
#         '''
#         if len(self.invariant_segs) == 0:
#             self.invariant_set(extended=False)

#         # Finds intersections between boundary and invariant set segments (boundary equilibria)
#         self.boundary_equilibria = []
#         self.interior_equilibria = []
#         for boundary_seg in self.boundary_segs:
#             for invariant_seg in self.invariant_segs:
                
#                 boundary_curve = geometry.LineString(boundary_seg)
#                 invariant_seg_curve = geometry.LineString(invariant_seg)
#                 intersections = intersection( boundary_curve, invariant_seg_curve )

#                 new_candidates = []
#                 if not intersections.is_empty:
#                     if hasattr(intersections, "geoms"):
#                         for geo in intersections.geoms:
#                             x, y = geo.xy
#                             x, y = list(x), list(y)
#                             new_candidates += [ [x[k], y[k]] for k in range(len(x)) ]
#                     else:
#                         x, y = intersections.xy
#                         x, y = list(x), list(y)
#                         new_candidates += [ [x[k], y[k]] for k in range(len(x)) ]
                
#                 for pt in new_candidates:

#                     eq_sol = self.optimize_over("boundary", init_x=pt)
#                     if (eq_sol) and "equilibrium" in eq_sol.keys():
#                         add_to(eq_sol, self.boundary_equilibria)

#                     eq_sol = self.optimize_over("interior", init_x=pt)
#                     if (eq_sol) and "equilibrium" in eq_sol.keys():
#                         add_to(eq_sol, self.interior_equilibria)

#                     branch_minimizer = self.optimize_over("min_branch", init_x=pt)
#                     if branch_minimizer and "type" in branch_minimizer.keys():
#                         add_to(branch_minimizer, self.branch_minimizers)

#                     branch_maximizer = self.optimize_over("max_branch", init_x=pt)
#                     if branch_maximizer and "type" in branch_maximizer.keys():
#                         add_to(branch_maximizer, self.branch_maximizers)

#         # self.branch_optimizers(verbose)

#         if verbose:
#             show_message(self.boundary_equilibria, "boundary equilibrium points")
#             show_message(self.interior_equilibria, "interior equilibrium points")

#             show_message(self.branch_minimizers, "branch minimizers")
#             show_message(self.branch_maximizers, "branch maximizers")

    # def equilibria(self, verbose=False):
    #     '''
    #     Computes all equilibrium points and local branch optimizers of the CLF-CBF pair, using the invariant set rectangular limits as initializers for the optimization algorithm.
    #     This method does not require the update of the complete invariant set geometry, 
    #     and is capable of computing the equilibrium points and local branch optimizers faster than the previous method.
    #     '''
    #     # Get initializers from boundary lines
    #     self.branch_initializers = [] 
    #     for line in self.limit_lines:
    #         self.branch_initializers += self.get_zero_det(line["x"], line["y"])

    #     # Find boundary, interior equilibria and branch optimizers
    #     self.boundary_equilibria = []
    #     self.interior_equilibria = []
    #     self.branch_minimizers = []
    #     self.branch_maximizers = []
    #     for pt in self.branch_initializers:

    #         eq_sol = self.optimize_over("boundary", init_x=pt)
    #         if (eq_sol) and "equilibrium" in eq_sol.keys():
    #             add_to(eq_sol, self.boundary_equilibria)

    #         eq_sol = self.optimize_over("interior", init_x=pt)
    #         if (eq_sol) and "equilibrium" in eq_sol.keys():
    #             add_to(eq_sol, self.interior_equilibria)

    #         branch_minimizer = self.optimize_over("min_branch", init_x=pt)
    #         if branch_minimizer and "type" in branch_minimizer.keys():
    #             add_to(branch_minimizer, self.branch_minimizers)

    #         branch_maximizer = self.optimize_over("max_branch", init_x=pt)
    #         if branch_maximizer and "type" in branch_maximizer.keys():
    #             add_to(branch_maximizer, self.branch_maximizers)

    #     # self.branch_optimizers(verbose)

    #     if verbose:
    #         show_message(self.boundary_equilibria, "boundary equilibrium points")
    #         show_message(self.interior_equilibria, "interior equilibrium points")

    #         show_message(self.branch_minimizers, "branch minimizers")
    #         show_message(self.branch_maximizers, "branch maximizers")

#     def branch_optimizers(self, verbose=False):
#         '''
#         Compute the branch optimizers
#         '''
#         self.connections_to_min = { i:[] for i in range(0,len(self.boundary_equilibria)) }
#         self.connections_to_max = { i:[] for i in range(0,len(self.boundary_equilibria)) }
#         self.branch_minimizers = []
#         self.branch_maximizers = []

#         # Create adjacency list for connections btw eq points and optimizers
#         for num_eq in range(len(self.boundary_equilibria)):
#             eq_sol = self.boundary_equilibria[num_eq]

#             branch_minimizer = self.optimize_over("min_branch", init_x=eq_sol["x"])
#             if branch_minimizer and "type" in branch_minimizer.keys():
#                 add_to(branch_minimizer, self.branch_minimizers, self.connections_to_min[num_eq])

#             branch_maximizer = self.optimize_over("max_branch", init_x=eq_sol["x"])
#             if branch_maximizer and "type" in branch_maximizer.keys():
#                 add_to(branch_maximizer, self.branch_maximizers, self.connections_to_max[num_eq])

#         # Checks if there exist removable optimizers
#         self.check_removables()

#         if verbose:
#             show_message(self.boundary_equilibria, "boundary equilibrium points")
#             show_message(self.interior_equilibria, "interior equilibrium points")

#             show_message(self.branch_minimizers, "branch minimizers")
#             show_message(self.branch_maximizers, "branch maximizers")

#             print(f"Connections to minimizers = {self.connections_to_min}")
#             print(f"Connections to maximizers = {self.connections_to_max}")

#     def check_removables(self):
#         '''
#         Checks if equilibrium point with index eq_index is removable.
#         Returns the corresponding minimizer/maximizer that removes the equilibrium point.
#         '''
#         self.min_removers, self.max_removers = [], []

#         for eq_index in range(len(self.boundary_equilibria)):
#             for minimizer_index in self.connections_to_min[eq_index]:
#                 for j in self.connections_to_min.keys():
#                     if j == eq_index:       # ignore if self
#                         continue
#                     if minimizer_index in self.connections_to_min[j] and np.linalg.norm( self.branch_minimizers[minimizer_index]["gradh"] ) > 1e-3:
#                         self.branch_minimizers[minimizer_index]["type"] = "remover"
#                         add_to(self.branch_minimizers[minimizer_index], self.min_removers)
#                         break

#             for maximizer_index in self.connections_to_max[eq_index]:
#                 for j in self.connections_to_max.keys():
#                     if j == eq_index:       # ignore if self
#                         continue
#                     if maximizer_index in self.connections_to_max[j] and np.linalg.norm( self.branch_maximizers[maximizer_index]["gradh"] ) > 1e-3:
#                         self.branch_maximizers[maximizer_index]["type"] = "remover"
#                         add_to(self.branch_maximizers[maximizer_index], self.max_removers)
#                         break

#     def optimize_over(self, optimization=None, **kwargs):
#         '''
#         Finds equilibrium points solutions using sliding mode control. If no initial point is specified, it selections a point at random from a speficied interval.
#         Returns a dict containing all relevant data about the found equilibrium point, including its stability.
#         '''
#         init_x_def = False
#         for key in kwargs.keys():
#             aux_key = key.lower()
#             if aux_key == "init_x":
#                 init_x = kwargs[key]
#                 init_x_def = True
#                 continue

#         if not init_x_def:
#             init_x = [ np.random.uniform( self.limits[k][0], self.limits[k][1] ) for k in range(self.n) ]

#         def invariant_set(var):
#             '''
#             Returns the vector residues of invariant set -> is zero for x in the invariant set
#             '''
#             x = var[0:self.n]
#             return det_invariant(x, self.kernel, self.P, self.cbf.Q, self.plant.get_F(), self.params)

#         def boundary_constraint(var):
#             '''
#             Returns the diff between mQm and 1
#             '''
#             x = var[0:self.n]
#             delta = var[self.n]

#             h = self.cbf.function(x)
#             return delta - np.abs(h)

#         def objective(var):
#             '''
#             Objective function to be minimized
#             '''
#             delta = var[self.n]
#             x = var[0:self.n]
            
#             if optimization == "boundary":
#                 return delta**2
#             elif optimization == "interior":
#                 return self.compute_lambda(x.tolist())**2
#             elif optimization == "min_branch":
#                 return self.cbf.function(x)
#             elif optimization == "max_branch":
#                 return -self.cbf.function(x)
#             else: 1.0

#         init_delta = 1.0
#         init_var = init_x + [init_delta]

#         constraints = [ {"type": "eq", "fun": invariant_set} ]
#         if optimization == "boundary":
#             constraints.append({"type": "ineq", "fun": boundary_constraint})

#         sol = minimize(objective, init_var, constraints=constraints)

#         eq_coords = sol.x[0:self.n].tolist()
#         l = self.compute_lambda(eq_coords)
#         h = self.cbf.function(eq_coords)
#         gradh = self.cbf.gradient(eq_coords)

#         sol_dict = None

#         # Valid solution is a point in the invariant set with lambda >= 0
#         if l >= 0 and np.abs(invariant_set(sol.x)) < 1e-3:
#             sol_dict = {}
#             sol_dict["x"] = eq_coords
#             sol_dict["lambda"] = l
#             sol_dict["delta"] = sol.x[self.n]
#             sol_dict["invariant_cost"] = invariant_set(sol.x)
#             sol_dict["h"] = h
#             sol_dict["gradh"] = np.linalg.norm(gradh)
#             sol_dict["init_x"] = init_x
#             # sol_dict["message"] = sol.message
        
#         # Boundary equilibrium point - compute stability
#         if (sol_dict) and (np.abs(sol_dict["h"]) <= 1e-3):
#             stability, eta = self.compute_stability(eq_coords, "boundary")
#             sol_dict["eta"], sol_dict["stability"] = eta, stability
#             sol_dict["equilibrium"] = "stable"
#             if stability > 0:
#                 sol_dict["equilibrium"] = "unstable"

#         # Interior equilibrium points (for now, stability is not computed)
#         if (sol_dict) and (optimization == "interior") and (np.abs(sol_dict["lambda"]) <= 1e-5):
#             stability, eta = self.compute_stability(eq_coords, "interior")
#             sol_dict["eta"], sol_dict["stability"] = eta, stability
#             sol_dict["equilibrium"] = "stable"
#             if stability > 0:
#                 sol_dict["equilibrium"] = "unstable"

#         # Minimizers
#         if (sol_dict) and optimization == "min_branch":
#             if sol_dict["gradh"] < 1e-03:
#                 sol_dict["type"] = "cbf_minimum"
#             else: sol_dict["type"] = "undefined"

#         # Maximizers
#         if (sol_dict) and optimization == "max_branch":
#             if sol_dict["h"] > 1e+05 or sol_dict["gradh"] > 1e+06:
#                 sol_dict = None                 # filters unbounded maximizers
#             else: sol_dict["type"] = "undefined"

#         return sol_dict

    # def get_zero_det(self, xg, yg):
    #     '''
    #     Returns the points where the determinant is zero over a 1D array with coords given by xg, yg 
    #     '''
    #     det_grid = self.det_invariant(xg, yg, extended=True)
    #     indexes = np.where(np.sign(det_grid[:-1]) != np.sign(det_grid[1:]))[0] + 1

    #     pts = []
    #     for i in indexes:
    #         pts.append( [xg[i][0], yg[i][0]] )
    #     return pts
    
