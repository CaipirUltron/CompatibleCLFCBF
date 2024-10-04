import logging
import itertools
import numpy as np
import sympy as sym
import operator

from copy import deepcopy
from common import *

'''
TODOs: 
i) transfer all functionality of KernelQuadratic to MultiPoly, including fitting.
To achieve this, it is necessary to convert the linear representation of MultiPoly into SOS representation (and vice-versa).
Since this is computationally expensive, put the functionality inside a callable method 
(the methods sos_kernel, sos_index_matrix and shape_matrix already have all the needed logic)
ii) make MultiPoly a child of Function (specially to use get_levels)
''' 

class MultiPoly:
    '''
    Class representing multivariable polynomials.
    To be used as a tool for the Kernel, KernelLinear and KernelQuadratic classes,
    representing polynomial data and performing operations between polynomials of any dimension/degree.
    Implementation: coefficients are ALWAYS float/np.ndarray (internally, they can be numeric or symbolic)
    '''
    def __init__(self, kernel: list, coeffs: list):

        if len(kernel) != len(coeffs):
            raise TypeError("Kernel and coefficients must have the same length.")
        self.kernel_dim = len(kernel)
        
        if not all([ len(mon) == len(kernel[0]) for mon in kernel ]):
            raise TypeError("All monomials in kernel must have the same number of dimensions.")
        self.n = len(kernel[0])

        if not all([ isinstance( coef, type(coeffs[0])) for coef in coeffs ]):
            raise TypeError("Coefficients must be of the same data type.")

        # If coefficients are lists, convert all to ndarray
        if isinstance(coeffs[0], list):
            coeffs = [ np.array(coef) for coef in coeffs ]

        self.kernel = kernel
        self.coeffs = coeffs

        # Classify number of dimensions and shape according to coefficients
        s_coef = self.coeffs[0]
        self.coeffs_type = type(s_coef)
        if isinstance(s_coef, (int, float)):
            self.ndim = 0
            self.shape = None
        elif isinstance(s_coef, np.ndarray):
            self.ndim = s_coef.ndim
            self.shape = s_coef.shape
        elif isinstance(s_coef, sym.Expr):
            self.ndim = 0
            self.shape = None
            if isinstance(s_coef, sym.MatrixExpr):
                self.ndim = 2
                self.shape = s_coef.shape

    def sort_kernel(self):
        '''
        Sorts the kernel monomials by total degree and ordering of variables.
        PS: computationally expensive, use it with care. 
        ''' 
        def degree_order(mon_and_coef):
            '''
            For sorting by monomial degree.
            Example in two dimensions:

            [(0,0)] -> monomials up to degree 0 ,
            [(1,0), (0,1)] -> monomials up to degree 1
            [(2,0), (1,1), (0,2)] -> monomials up to degree 2

            Then, (0,0), (1,0), (0,1), (2,0), (1,1), (0,2) would be the correct 
            order for polynomials in two dimensions and up to degree 2
            (groups of monomials with lower maximum degree come first).
            '''
            mon = mon_and_coef[0]
            # coeff = mons_and_coefs[1]
            return sum(mon)

        def pos_order(mon_and_coef):
            '''
            For sorting by exponent dimensional order (favoring lower indexed dimensions).
            Example in two dimensions:
            (0,0), (1,0), (0,1), (2,0), (1,1), (0,2) would be the correct order 
            for polynomials in two dimensions and up to degree 2 (higher exponents 
            in lower indexed dimensions comes first).
            '''
            mon = mon_and_coef[0]
            # coeff = mon_and_coef[1]
            return sum([ mon[dim]*(2**dim) for dim in range(len(mon)) ])

        mons_and_coefs = list(zip(self.kernel, self.coeffs))
        mons_and_coefs.sort(key=degree_order)

        self.kernel, self.coeffs = [], []
        for degree, mons_and_coefs_by_degree in itertools.groupby(mons_and_coefs, degree_order):

            mons_and_coefs_by_degree_list = list(mons_and_coefs_by_degree)
            mons_and_coefs_by_degree_list.sort(key=pos_order)

            for mon, coef in mons_and_coefs_by_degree_list:
                self.kernel.append( mon )
                self.coeffs.append( coef )

    def is_empty(self):
        ''' Checks if polynomial is empty '''
        return len(self.kernel) == 0 and len(self.coeffs)

    def set_coef(self, monomials, coeffs):
        '''
        Sets the coefficients corresponding to monomials.
        '''
        if isinstance(monomials, list) and isinstance(coeffs, list):
            if len(coeffs) != len(monomials):
                raise TypeError("Coefficients and corresponding monomials do not match.")            
        else:
            monomials = [ monomials ]
            coeffs = [ coeffs ]

        for mon, coef in zip(monomials, coeffs):

            if len(mon) != self.n:
                raise TypeError("Invalid monomial dimensions.")
            
            if np.any(np.array(mon) < 0):
                raise NotImplementedError("MultiPoly does not support negative exponentials.")

            if isinstance(coef, np.ndarray) and coef.shape != self.shape:
                raise TypeError("Invalid coefficient shape.")
            
            if mon not in self.kernel:
                self.kernel.append(mon)
                self.coeffs.append(coef)
            else:
                index = self.kernel.index(mon)
                self.coeffs[index] = coef

    def get_coef(self, monomials):
        '''
        Gets the coefficients corresponding to the given monomials.
        '''
        if not isinstance(monomials, list):
            monomials = [ monomials ]

        coefs = [ None for _ in monomials ]
        for k, mon in enumerate(monomials):

            if len(mon) != self.n:
                raise TypeError("Invalid monomial dimensions.")
            
            if mon in self.kernel:
                index = self.kernel.index(mon)
                coefs[k] = self.coeffs[index]

        if len(coefs) > 1:
            return coefs
        else:
            return coefs[0]

    def _addition(self, poly, op):
        '''
        Add/subtract two instances of multipoly.
        op = +1 sums, op = -1 subtracts
        '''
        if self.shape != poly.shape:
            raise NotImplementedError("Cannot sum polynomials of different shapes.")

        if np.abs(op) != 1:
            raise ValueError("\"op\" argument should be +1 or -1")

        # Initialize kernel of the resulting polynomial.
        res_kernel = list( set(self.kernel).union(set(poly.kernel)) )

        # Initialize coefficients of the resulting polynomial with zeros.
        if self.ndim == 0: res_coeffs = [ 0.0 for _ in res_kernel ]
        else: res_coeffs = [ np.zeros(self.shape) for _ in res_kernel ]

        # Sums/subtracts the coefficients corresponding to each monomial  
        for mon in res_kernel:
            coef_index = res_kernel.index( mon )

            if mon in self.kernel:
                i = self.kernel.index( mon )
                res_coeffs[coef_index] = res_coeffs[coef_index] + self.coeffs[i]

            if mon in poly.kernel:
                i = poly.kernel.index( mon )
                res_coeffs[coef_index] = res_coeffs[coef_index] + op * poly.coeffs[i]

        return MultiPoly(kernel=res_kernel, coeffs=res_coeffs)

    def _add(self, poly):
        ''' Adds two instances of multipoly '''
        return self._addition(poly, +1)

    def _sub(self, poly):
        ''' Subtract two instances of multipoly '''
        return self._addition(poly, -1)

    def _multiply(poly1, poly2, op):
        '''
        Polynomial multiplication (term by term or matrix-like).
        op = +1 for term-by-term, op = -1 for matrix-like
        '''
        if np.abs(op) != 1:
            raise ValueError("\"op\" argument should be +1 or -1")

        if op > 0: sample_term = poly1.coeffs[0] * poly2.coeffs[0]
        if op < 0: sample_term = poly1.coeffs[0] @ poly2.coeffs[0]
        
        if hasattr(sample_term, "ndim") and sample_term.ndim == 0: 
            sample_term = float(sample_term)

        mons_and_coefs1 = zip(poly1.kernel, poly1.coeffs)
        mons_and_coefs2 = zip(poly2.kernel, poly2.coeffs)

        res_kernel, res_coeffs = [], []
        for (z1, z2) in itertools.product( mons_and_coefs1, mons_and_coefs2 ):

            mon1, coeff1 = z1[0], z1[1]
            mon2, coeff2 = z2[0], z2[1]

            mon = tuple([ int(dim1)+int(dim2) for dim1, dim2 in zip(mon1, mon2) ])
            if mon not in res_kernel:
                res_kernel.append(mon)
                if isinstance(sample_term, float):
                    res_coeffs.append(0.0)
                if isinstance(sample_term, np.ndarray): 
                    res_coeffs.append(np.zeros(sample_term.shape))
            
            index = res_kernel.index( mon )

            if op > 0: 
                res_coeffs[index] += coeff1 * coeff2       # scalar multiplication
            if op < 0: 
                res_coeffs[index] += coeff1 @ coeff2       # matrix multiplication

        return MultiPoly(kernel=res_kernel, coeffs=res_coeffs)

    def _mul(poly1, poly2):
        ''' Term by term polynomial multiplication '''
        return MultiPoly._multiply(poly1, poly2, +1)

    def _matmul(poly1, poly2):
        ''' Matrix polynomial multiplication '''
        return MultiPoly._multiply(poly1, poly2, -1)

    def _operator_fallbacks(operation, op_name):
        ''' Implementation of forward, reverse and inplace operations for MultiPoly '''

        def not_multipoly(self, other):
            ''' What to do if the second operand is not a MultiPoly object '''

            new_poly = deepcopy(self)
            
            if op_name == operator.add or op_name == operator.sub:
                curr_const_coef = self.get_coef( (0,0) )
                new_poly.set_coef( (0,0), op_name(curr_const_coef, other) )

            if op_name == operator.mul or op_name == operator.matmul:
                new_poly.set_coef( new_poly.kernel, [ op_name(coef, other) for coef in new_poly.coeffs ] )
            
            return new_poly

        def forward(self, other):

            if isinstance(other, MultiPoly):
                ''' Forward operation when right term is a MultiPoly '''
                return operation(self, other)
            
            elif isinstance(other, (int, float, np.ndarray)):
                ''' Forward operation when right term is not a MultiPoly '''
                return not_multipoly(self, other)
                            
            else: return NotImplemented
            
        forward.__name__ = '__' + op_name.__name__ + '__'
        forward.__doc__ = operation.__doc__

        def reverse(self, other):

            if isinstance(other, MultiPoly):
                ''' Reverse operation when right term is a MultiPoly '''
                return operation(other, self)
            
            elif isinstance(other, (int, float, np.ndarray)):
                print('AHA')
                ''' Reverse operation when right term is not a MultiPoly '''
                return not_multipoly(self, other)
            
            else: return NotImplemented
            
        reverse.__name__ = '__r' + op_name.__name__ + '__'
        reverse.__doc__ = operation.__doc__

        def inplace(self, other):
            
            if isinstance(other, MultiPoly):
                ''' Inplace operation when right term is a MultiPoly '''
                return operation(self, other)
            
            elif isinstance(other, (int, float, np.ndarray)):
                ''' Inplace operation when right term is not a MultiPoly '''
                return not_multipoly(self, other)
            
            else: return NotImplemented
            
        inplace.__name__ = '__i' + op_name.__name__ + '__'
        inplace.__doc__ = operation.__doc__

        return forward, reverse, inplace

    __add__, __radd__, __iadd__ = _operator_fallbacks(_add, operator.add)
    __sub__, __rsub__, __isub__ = _operator_fallbacks(_sub, operator.sub)
    __mul__, __rmul__, __imul__ = _operator_fallbacks(_mul, operator.mul)
    __matmul__, __rmatmul__, __imatmul__ = _operator_fallbacks(_matmul, operator.matmul)

    def __pow__(self, power):
        '''
        Polynomial exponentiation (integer exponents only).
        If self.ndim is 0 or 1, multiplicative exponentiation is performed.
        If self.ndim is 2, matrix exponentiation is performed instead.
        '''
        if not isinstance(power, int):
            raise NotImplementedError("Only raise a matrix to an integer power")
        
        if ( self.ndim == 1 and self.shape != (1,) ) or ( self.ndim == 2 and self.shape[0] != self.shape[1] ):
            raise NotImplementedError("Can only raise a polynomial to a power if is a scalar or square matrix.")

        if self.ndim == 0:
            new_poly = MultiPoly(kernel=[ tuple( 0 for _ in range(self.n) ) ], coeffs = [ 1.0 ])
        if self.ndim == 1:
            new_poly = MultiPoly(kernel=[ tuple( 0 for _ in range(self.n) ) ], coeffs = [ np.ones(self.shape) ])
        if self.ndim == 2:
            new_poly = MultiPoly(kernel=[ tuple( 0 for _ in range(self.n) ) ], coeffs = [ np.eye(self.shape[0]) ])

        for _ in range(power):
            if self.ndim != 2: 
                new_poly = new_poly * self
            else: 
                new_poly = new_poly @ self

        return new_poly

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

    def __call__(self, x):
        ''' Computes polynomial value '''

        if len(x) != self.n:
            raise TypeError("Input has incorrect dimensions.")

        s = np.zeros(self.shape)
        for mon, coeff in zip(self.kernel, self.coeffs):
            s += coeff * np.prod([ x[i]**power for i, power in enumerate(mon) ])

        return s

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
        return self.__call__(x)

    def filter(self):
        '''
        Deletes entries of self.coeffs and self.kernel with zero coefficients.
        '''
        to_be_deleted = []
        for i, coef in enumerate(self.coeffs):

            if isinstance(coef, sym.Expr) and coef == sym.S.Zero:
                to_be_deleted.append(i)

            if isinstance(coef, sym.MatrixExpr) and coef == sym.ZEROS(*coef.shape):
                to_be_deleted.append(i)

            if isinstance(coef, (int, np.ndarray)) and np.linalg.norm(coef) < 1e-12:
                to_be_deleted.append(i)

        for i in to_be_deleted:
            del self.coeffs[i]
            del self.kernel[i]

    def hstack(poly1, poly2):
        ''' Horizontally stack two polys. Is able to stack empty polynomials '''

        if poly1.is_empty():
            return poly2

        if poly2.is_empty():
            return poly1

        if poly1.kernel != poly2.kernel:
            raise NotImplementedError("Only polynomials with the same kernel can be vertically stacked.")

        if poly1.shape[0] != poly2.shape[0]:
            raise TypeError("Only polynomials with the same number of lines can be horizontally stacked.")

        if poly1.coeffs_type is sym.MatrixExpr:
            index_coeffs = [ sym.BlockMatrix([c1,c2]) for c1, c2 in zip(poly1.coeffs, poly2.coeffs) ]
        else:
            index_coeffs = [ np.hstack([c1,c2]) for c1, c2 in zip(poly1.coeffs, poly2.coeffs) ]
            
        return MultiPoly(poly1.kernel, index_coeffs)

    def vstack(poly1, poly2):
        ''' Vertically stack two polys. Is able to stack empty polynomials '''

        if poly1.is_empty():
            return poly2

        if poly2.is_empty():
            return poly1

        if poly1.kernel != poly2.kernel:
            raise NotImplementedError("Only polynomials with the same kernel can be vertically stacked.")

        if poly1.shape[1] != poly2.shape[1]:
            raise TypeError("Only polynomials with the same number of columns can be vertically stacked.")

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

        transposed_coeffs = self.coeffs
        if self.ndim == 2:
            transposed_coeffs = [ coeff.T for coeff in self.coeffs ]            

        return MultiPoly( kernel=self.kernel, coeffs=transposed_coeffs )

    @classmethod
    def empty(cls):
        '''
        Returns an empty polynomial.
        '''
        return cls(kernel=[],coeffs=[])
    
    @classmethod
    def constant(cls, const: float, **kwargs):
        '''
        Returns a constant polynomial with specified dimension, kernel and shape.
        If no dimension is passed, it is initialized with dim = 2.
        If no kernel is passed, it is initialized with a constant kernel = [(0,0,...,0)] of that given dimension.
        If no shape is passed, a scalar polynomial is returned.
        '''
        dim = 2
        shape = None
        kernel = [ tuple( 0 for _ in range(dim) ) ]

        for key in kwargs.keys():
            if key.lower() == "dim":
                dim = kwargs["dim"]
                continue
            if key.lower() == "shape":
                shape = kwargs["shape"]
                continue
            if key.lower() == "kernel":
                kernel = kwargs["kernel"]
                continue

        # If no shape is passed, returns a scalar polynomial
        if shape == None:
            return cls(kernel=kernel, coeffs=[ const for _ in kernel ])
        
        # If shape is passed, returns a zero polynomial with the given shape
        return cls(kernel=kernel, coeffs=[ const*np.ones(shape) for _ in kernel ])
    
    @classmethod
    def zeros(cls, **kwargs):
        return MultiPoly.constant(const=0.0, **kwargs)
    
    @classmethod
    def ones(cls, **kwargs):
        return MultiPoly.constant(const=1.0, **kwargs)