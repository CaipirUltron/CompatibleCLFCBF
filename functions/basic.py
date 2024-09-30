import logging
import itertools
import numpy as np
import sympy as sym

import operator
import contourpy as ctp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from abc import ABC, abstractmethod
from dataclasses import dataclass

from common import *
from dynamic_systems import Integrator

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

@dataclass
class LeadingShape:
    ''' Data class for a leading shape (to be used as an approximation tool) '''
    shape: np.ndarray
    bound: str = ''
    approximate: bool = False

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

        if self._output_dim == 1: self.generate_contour()

    def _validate(self, point):
        ''' Validates input data '''
        if not isinstance(point, (list, tuple, np.ndarray)): raise Exception("Input data point is not a numeric array.")
        if isinstance(point, (list, tuple)): point = np.array(point)
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

class MultiPoly:
    '''
    Class representing multivariable polynomials.
    To be used as a tool for the Kernel, KernelLinear and KernelQuadratic classes,
    simply to represent polynomial data and perform operations between polynomials.
    Implementation: coefficients are ALWAYS float/np.ndarray (internally, they can be numeric or symbolic)
    '''
    def __init__(self, kernel, coeffs = None):

        self.kernel = kernel
        self.coeffs = coeffs
        self.coeffs_type = None       # scalar, vector or matrix

        self.is_empty = False
        if self.kernel == None:
            self.is_empty = True
            return

        if self.coeffs is None:
            self.coeffs = [ None for _ in self.kernel ]

        # Convert to standard types
        for k, coeff in enumerate(self.coeffs):
            if isinstance(coeff, int):
                self.coeffs[k] = float(coeff)
                self.coeffs_type = float
            elif isinstance(coeff, list):
                self.coeffs[k] = np.array(coeff)
                self.coeffs_type = np.ndarray
            elif isinstance(coeff, np.ndarray):
                self.coeffs_type = np.ndarray
            elif isinstance(coeff, sym.Expr):
                self.coeffs_type = sym.Expr
                if isinstance(coeff, sym.MatrixExpr):
                    self.coeffs_type = sym.MatrixExpr

        if self.coeffs_type is np.ndarray:
            self.shape = self.coeffs[0].shape
            self.ndim = self.coeffs[0].ndim
        elif self.coeffs_type is sym.MatrixExpr:
            self.shape = self.coeffs[0].shape
            if self.shape[0] == 1 and self.shape[1] == 1:
                self.ndim = 0
            elif self.shape[0] == 1 or self.shape[1] == 1:
                self.ndim = 1
            else:
                self.ndim = 2
        else:
            self.ndim = 0
            self.shape = ()

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
        self.n = len(self.kernel[0])

    def _verify_op(poly1, poly2):
        ''' Checks if aritmetic operations can be executed '''

        type1 = np.array(poly1.kernel).dtype
        type2 = np.array(poly2.kernel).dtype
        if (type1, type2) != ('int','int'):
            raise Exception("Monomial exponents must be integers.")

    def _addition(poly1, poly2, op):
        '''
        Add/subtract two instances of multipoly.
        op = +1 sums, op = -1 subtracts
        '''
        MultiPoly._verify_op(poly1, poly2)

        res_kernel = list( set(poly1.kernel).union(set(poly2.kernel)) )

        res_coeffs = []
        for mon in res_kernel:

            if mon in poly1.kernel:
                i = poly1.kernel.index( mon )
                new_c1 = poly1.coeffs[i]

            if mon in poly2.kernel:
                i = poly2.kernel.index( mon )
                if op > 0: 
                    new_c2 = +poly2.coeffs[i]
                else: 
                    new_c2 = -poly2.coeffs[i]

            res_coeffs.append( new_c1 + new_c2 )

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
        res_coeffs = [ None for _ in range(len(res_kernel)) ]

        # Populate coefficients
        zipped1 = zip(poly1.kernel, poly1.coeffs)
        zipped2 = zip(poly2.kernel, poly2.coeffs)
        for (z1, z2) in itertools.product( zipped1, zipped2 ):

            mon1, coeff1 = z1[0], z1[1]
            mon2, coeff2 = z2[0], z2[1]

            mon = tuple([ int(dim1)+int(dim2) for dim1, dim2 in zip(mon1,mon2) ])
            index = res_kernel.index( mon )

            if op > 0: term_to_be_added = coeff1 * coeff2       # scalar multiplication
            else: term_to_be_added = coeff1 @ coeff2            # matrix multiplication

            if res_coeffs[index] is None:
                res_coeffs[index] = term_to_be_added
            else:
                res_coeffs[index] += term_to_be_added

        return MultiPoly(kernel=res_kernel, coeffs=res_coeffs)

    def _mul(poly1, poly2):
        ''' Term by term polynomial multiplication '''
        return MultiPoly._multiply(poly1, poly2, +1)

    def _matmul(poly1, poly2):
        ''' Matrix polynomial multiplication '''
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

    def __pos__(self):
        return self

    def __neg__(self):
        index_coeffs = [ -c for c in self.coeffs ]
        return MultiPoly(self.kernel, index_coeffs)

    def __getitem__(self, items):
        ''' Subscritable method '''

        if not isinstance(items, (int, tuple)):
            raise IndexError

        if isinstance(items[0], slice):
            if items[0].start == items[0].stop or items[0].start == self.shape[0]:
                return MultiPoly.empty()
            
        if isinstance(items[1], slice):
            if items[1].start == items[1].stop or items[1].start == self.shape[1]:
                return MultiPoly.empty()

        index_coeffs = [ c[items] for c in self.coeffs ]
        return MultiPoly(self.kernel, index_coeffs)

    def __repr__(self):
        ''' Representation of MultiPoly '''

        if self.is_empty:
            return "Empty polynomial"

        if self.ndim == 0: type_text = "scalar"
        elif self.ndim == 1: type_text = "vector"
        elif self.ndim == 2: type_text = "matrix"

        poly_repr = f"{type_text.capitalize()} poly on x:\n"
        for coeff, power in zip(self.coeffs, self.kernel):
            if isinstance(coeff, (int, float)):
                if coeff > 0: sign_text = "+ "
                else: sign_text = "- "
                poly_repr += sign_text + f"{abs(coeff):.3f}*x^{power} "
            else:
                poly_repr +=  f"( {coeff} )*x^{power} + "

        return poly_repr

    def __str__(self):
        ''' Printing for MultiPoly '''
        return self.__repr__()

    def polyder(self):
        ''' Get the polynomial derivatives with respect to all variables '''

        poly_diffs = []
        EYE = np.eye(self.n, dtype=int)
        for i in range(self.n):
            diff_coeffs = []
            diff_kernel = []
            for k, mon in enumerate(self.kernel):
                coeff = self.coeffs[k]
                new_mon = np.array(mon) - EYE[i,:]
                if new_mon[i] >= 0:
                    if tuple(new_mon) not in diff_kernel:
                        diff_kernel.append( tuple(new_mon) )
                        diff_coeffs.append( coeff * mon[i] )
                    else:
                        id = diff_kernel.index( tuple(new_mon) )
                        diff_coeffs[id] += coeff * mon[i]
            poly_diffs.append( MultiPoly(kernel=diff_kernel, coeffs=diff_coeffs) )

        return poly_diffs

    def polyval(self, x):
        ''' Computes polynomial value '''

        if len(x) != self.n:
            raise Exception("Input has incorrect dimensions.")

        s = np.zeros(self.shape)
        for mon, coeff in zip(self.kernel, self.coeffs):
            s += coeff * np.prod([ x[i]**power for i, power in enumerate(mon) ])

        return s

    def filter(self):
        ''' Returns a new multipoly without zero coefficient terms '''

        new_kernel, new_coeffs = [], []
        for mon, coeff in zip(self.kernel, self.coeffs):

            to_be_added = False
            if isinstance(coeff, float) and coeff > 0.0:
                to_be_added = True
            elif isinstance(coeff, np.ndarray):
                if coeff.dtype in ("int", "float") and np.linalg.norm(coeff) > 0.0:
                    to_be_added = True
                if coeff.dtype == "object" and np.all(coeff != np.zeros(coeff.shape)):
                    to_be_added = True

            if to_be_added:
                new_kernel.append(mon)
                new_coeffs.append(coeff)

        return MultiPoly(new_kernel, new_coeffs)

    def hstack(poly1, poly2):
        ''' Horizontally stack two polys. Is able to stack empty polynomials '''

        if poly1.is_empty:
            return poly2

        if poly2.is_empty:
            return poly1

        if poly1.kernel != poly2.kernel:
            raise TypeError("LIMITATION: polynomials must have the same kernel.")

        if poly1.shape[0] != poly2.shape[0]:
            raise TypeError("Cannot hor. stack polynomials with diff. number of lines.")

        if poly1.coeffs_type is sym.MatrixExpr:
            index_coeffs = [ sym.BlockMatrix([c1,c2]) for c1, c2 in zip(poly1.coeffs, poly2.coeffs) ]
        else:
            index_coeffs = [ np.hstack([c1,c2]) for c1, c2 in zip(poly1.coeffs, poly2.coeffs) ]
            
        return MultiPoly(poly1.kernel, index_coeffs)

    def vstack(poly1, poly2):
        ''' Vertically stack two polys. Is able to stack empty polynomials '''

        if poly1.is_empty:
            return poly2

        if poly2.is_empty:
            return poly1

        if poly1.kernel != poly2.kernel:
            raise TypeError("LIMITATION: polynomials must have the same kernel.")

        if poly1.shape[1] != poly2.shape[1]:
            raise TypeError("Cannot vert. stack polynomials with diff. number of lines.")

        if poly1.coeffs_type is sym.MatrixExpr:
            index_coeffs = [ sym.BlockMatrix([[c1],[c2]]) for c1, c2 in zip(poly1.coeffs, poly2.coeffs) ]
        else:
            index_coeffs = [ np.vstack([c1,c2]) for c1, c2 in zip(poly1.coeffs, poly2.coeffs) ]

        return MultiPoly(poly1.kernel, index_coeffs)

    def minor(self, index: tuple):
        ''' Returns the cofactor of index '''

        if self.ndim != 2:
            raise TypeError("Cannot compute the minor of a non-matrix polynomial.")
        
        i,j = index            

        blk11 = self[0:i , 0:j]
        blk12 = self[0:i ,j+1:]
        blk21 = self[i+1:, 0:j]
        blk22 = self[i+1:,j+1:]

        blk1 = MultiPoly.hstack(blk11, blk12)
        blk2 = MultiPoly.hstack(blk21, blk22)

        minor = MultiPoly.vstack(blk1, blk2)
        return minor

    def determinant(self):
        '''
        Returns the polynomial determinant for a multiply matrix.
        The algorithm used here is Laplace expansion with recursion.
        '''
        if self.shape[0] != self.shape[1]:
            raise TypeError("Cannot compute the determinant of a non-square matrix polynomial.")

        if self.ndim == 0 or self.shape == (1,1):   # polynomial is a scalar
            return self
        
        i = 1           # expansion from first line
        det = None
        for j in range(self.shape[1]):
            term = (-1)**(i + j) * self[i, j] * self.minor( (i,j) ).determinant()
            if det is None: det = term
            else: det += term

        return det.scalar()

    def sos_kernel(self):
        ''' Function for computing the corresponding polynomial SOS kernel '''

        sos_kernel = []
        for mon in self.kernel:
            possible_curr_combinations = set([ tuple(np.array(mon1)+np.array(mon2)) for mon1,mon2 in itertools.combinations(sos_kernel, 2) ])

            if mon in possible_curr_combinations: 
                continue

            if len(possible_curr_combinations) == 0:
                sos_kernel.append(mon)
                continue

            # If mon is not on possible with current combinations, check if its possible to create it from them...
            possibilities = []

            # If all exponents of mon are even, it can be created from 
            if np.all([ exp % 2 == 0 for exp in mon ]):
                possibilities.append( tuple([int(exp/2) for exp in mon]) )

            # Checks if mon can be created from the combination of monomials already in self._sos_monomials and another
            for sos_mon in sos_kernel:
                pos = np.array(mon) - np.array(sos_mon)
                if np.all(pos >= 0): 
                    possibilities.append( tuple([ int(exp) for exp in pos ]) )

            index = np.argmin([ np.linalg.norm(pos) for pos in possibilities ])
            new_sos_mon = possibilities[index]
            if new_sos_mon not in sos_kernel:
                sos_kernel.append(new_sos_mon)

        return sos_kernel

    def sos_index_matrix(self, sos_kernel):
        '''
        Computes the index matrix representing the rule for placing the coefficients in the correct places on the 
        shape matrix of the SOS representation. Algorithm gives preference for putting the elements of coeffs 
        closer to the main diagonal of the SOS matrix.
        '''     
        sos_kernel_dim = len(sos_kernel)
        index_matrix = -np.ones([sos_kernel_dim, sos_kernel_dim], dtype='int')

        for k in range(len(self.kernel)):

            mon = self.kernel[k]

            # Checks the possible (i,j) locations on SOS matrix where the monomial can be put
            possible_places = []
            for (i,j) in itertools.product(range(sos_kernel_dim),range(sos_kernel_dim)):
                if i > j: continue
                sos_mon_i, sos_mon_j = np.array(sos_kernel[i]), np.array(sos_kernel[j])

                if mon == tuple(sum([sos_mon_i, sos_mon_j])):
                    possible_places.append( (i,j) )

            # From these, chooses the place closest to SOS matrix diagonal
            distances_from_diag = np.array([ np.abs(place[0] - place[1]) for place in possible_places ])
            i,j = possible_places[np.argmin(distances_from_diag)]

            index_matrix[i,j] = k

        return index_matrix

    def shape_matrix(self, sos_kernel, sos_index_matrix):
        '''
        Using the index matrix, returns the SOS shape matrix correctly populated by the coefficients.
        '''
        if len(self.coeffs) != len(self.kernel):
            raise Exception("The number of coefficients must be equal to the kernel dimension.")
        
        sos_kernel_dim = len(sos_kernel)
        shape_matrix = np.zeros([sos_kernel_dim, sos_kernel_dim]).tolist()

        for (i,j) in itertools.product(range(sos_kernel_dim),range(sos_kernel_dim)):
            if i > j: continue

            k = sos_index_matrix[i,j]
            if k >= 0:
                if i == j:
                    shape_matrix[i][j] = self.coeffs[k]
                else:
                    shape_matrix[i][j] = 0.5 * self.coeffs[k]
                    shape_matrix[j][i] = 0.5 * self.coeffs[k]

        return shape_matrix

    def scalar(self):
        ''' Convert 1x1 matrix poly to scalar poly '''

        if self.shape != (1,1):
            logging.warning("Only 1x1 matrix polys can be converted to scalar.")
            return self
        
        return MultiPoly(self.kernel, coeffs = [ c[0,0] for c in self.coeffs ])

    def hyperbolic_transform(self):

        if self.shape not in ( (1,1), () ):
            raise Exception("Hyperbolic transform is only defined for scalar polynomials.")
        
        ncoeffs = len(self.coeffs)
        new_variables = []
        matrix = np.array([ [] for _ in range(ncoeffs) ])
        for k, coeff in enumerate(self.coeffs):
            
            for term in sym.Add.make_args(coeff):
                
                for atom in term.args:
                    constant = 1.0
                    if atom != atom.free_symbols:
                        constant = atom
                        break

                term = term/constant
                if term not in new_variables:
                    new_variables.append(term)
                    matrix = np.hstack([matrix, np.zeros((ncoeffs, 1))])
                    matrix[k,-1] = constant
                else:
                    term_index = new_variables.index( term )
                    matrix[k, term_index] = constant

        return new_variables, matrix

        print(f"New hyperbolic variables = ")
        for var in new_variables: print(var)
        print(f"Transformation matrix = \n{matrix}")
        # print( np.linalg.lstsq( matrix, np.array([ 1.0 if k == 0 else 0.0 for k in range(ncoeffs) ]), rcond=None ) )

    @property
    def T(self):
        ''' Transpose operation, defined for array-like coefficients '''

        if self.ndim == 2:
            transposed_coeffs = [ coeff.T for coeff in self.coeffs ]
        if self.ndim == 0 or self.ndim == 1:
            transposed_coeffs = self.coeffs

        return MultiPoly( kernel=self.kernel, coeffs=transposed_coeffs )

    @classmethod
    def empty(cls):
        return cls(kernel=None)

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
    
