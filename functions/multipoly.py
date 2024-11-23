import logging
import itertools
import operator

import cvxpy as cvx
import numpy as np
import sympy as sym

from copy import deepcopy
from scipy.optimize import least_squares

from common import *
from functions.basic import Function
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyder, polyint

class Poly(Polynomial):
    '''
    Child of Polynomial with alternate call method for MultiPolys
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, arg):
        '''
        Modified call method to return a MultiPoly if arg is a MultiPoly
        '''
        if not isinstance(arg, MultiPoly):
            return super().__call__(arg)
        
        poly = MultiPoly.zeros(dim = arg.n, shape = (arg.shape))
        for k, c in enumerate(self.coef):
            poly += ( arg**k ) * c

        return poly

class MultiPoly(Function):
    '''
    Class representing multivariable polynomials.
    To be used as a tool for the Kernel, KernelLinear and KernelQuadratic classes,
    representing polynomial data and performing operations between polynomials of any dimension/degree.
    Implementation: coefficients are ALWAYS float/np.ndarray (internally, they can be numeric or symbolic)
    '''
    def __init__(self, kernel: list, coeffs: list, *args, **kwargs):
        
        if len(kernel) != len(coeffs):
            raise TypeError("Kernel and coefficients must have the same length.")
        self.kernel_dim = len(kernel)
        
        if not all([ len(mon) == len(kernel[0]) for mon in kernel ]):
            raise TypeError("All monomials in kernel must have the same number of dimensions.")
        
        if not all([ isinstance( coef, type(coeffs[0])) for coef in coeffs ]):
            raise TypeError("Coefficients must be of the same data type.")

        # If coefficients are lists, convert all to ndarray
        if isinstance(coeffs[0], list):
            coeffs = [ np.array(coef) for coef in coeffs ]

        self.kernel = kernel
        self.coeffs = coeffs

        self._poly_diff = None
        self._poly_grad = None
        self._poly_jaco = None
        self._poly_hess = None

        self.sos_kernel = None
        self.sos_indexes = None
        self.sos_locations = None

        # Classify number of dimensions and shape according to coefficients
        s_coef = self.coeffs[0]
        self.coeffs_type = type(s_coef)
        if isinstance(s_coef, int | float | complex):
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

        super().__init__(*args, **kwargs)

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

        # Setups output dimension
        self.n = self._dim
        self._output_dim = None
        if self.ndim == 0:
            self._output_dim = 1

        # Setups symbols for SOS decomposition
        state_symbols = ''
        for dim in range(self._dim):
            state_symbols += 'x'+str(dim+1)+' '
        self.symbols = sym.symbols(state_symbols)

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
        op = +1 for term-by-term, 
        op = -1 for matrix-like
        op = 0 for outer product
        '''
        if op not in (-1, 0, 1):
            raise ValueError("\"op\" argument should be -1, 0 or +1")

        if op > 0: sample_term = poly1.coeffs[0] * poly2.coeffs[0]
        if op < 0: sample_term = poly1.coeffs[0] @ poly2.coeffs[0]
        if op == 0: sample_term = np.outer( poly1.coeffs[0], poly2.coeffs[0] )
        
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
                res_coeffs[index] += coeff1 * coeff2                # scalar multiplication
            if op < 0: 
                res_coeffs[index] += coeff1 @ coeff2                # matrix multiplication
            if op == 0:
                res_coeffs[index] += np.outer( coeff1, coeff2 )     # outer product

        return MultiPoly(kernel=res_kernel, coeffs=res_coeffs)

    def _mul(poly1, poly2):
        ''' Term by term polynomial multiplication '''
        return MultiPoly._multiply(poly1, poly2, +1)

    def _matmul(poly1, poly2):
        ''' Matrix polynomial multiplication '''
        return MultiPoly._multiply(poly1, poly2, -1)

    def _operator_fallbacks(operation, op_name):
        ''' Implementation of forward, reverse and inplace operations for MultiPoly '''

        def not_multipoly(self, other, op):
            ''' What to do if the second operand is not a MultiPoly object '''

            if op_name == operator.add or op_name == operator.sub:

                new_poly = deepcopy(self)

                curr_const_coef = self.get_coef( (0,0) )
                if curr_const_coef == None:
                        curr_const_coef = np.zeros(self.shape)

                new_poly.set_coef( (0,0), op_name(curr_const_coef, other) )

            if op_name == operator.mul or op_name == operator.matmul:

                if op == 'forward' or 'inplace':
                    new_poly = MultiPoly(kernel=self.kernel, coeffs=[ op_name(coef, other) for coef in self.coeffs ])
                elif op == 'reverse':
                    new_poly = MultiPoly(kernel=self.kernel, coeffs=[ op_name(other, coef) for coef in self.coeffs ])
                else:
                    new_poly = NotImplemented
                
            return new_poly

        def forward(self, other):

            if isinstance(other, MultiPoly):
                ''' Forward operation when right term is a MultiPoly '''
                return operation(self, other)
            
            elif isinstance(other, (int, float, np.ndarray)):
                ''' Forward operation when right term is not a MultiPoly '''
                return not_multipoly(self, other, op='forward')
                            
            else: return NotImplemented
            
        forward.__name__ = '__' + op_name.__name__ + '__'
        forward.__doc__ = operation.__doc__

        def reverse(self, other):

            if isinstance(other, MultiPoly):
                ''' Reverse operation when right term is a MultiPoly '''
                return operation(other, self)
            
            elif isinstance(other, (int, float, np.ndarray)):
                ''' Reverse operation when right term is not a MultiPoly '''
                return not_multipoly(self, other, op='reverse')
            
            else: return NotImplemented
            
        reverse.__name__ = '__r' + op_name.__name__ + '__'
        reverse.__doc__ = operation.__doc__

        def inplace(self, other):
            
            if isinstance(other, MultiPoly):
                ''' Inplace operation when right term is a MultiPoly '''
                return operation(self, other)
            
            elif isinstance(other, (int, float, np.ndarray)):
                ''' Inplace operation when right term is not a MultiPoly '''
                return not_multipoly(self, other, op='inplace')
            
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

        if not isinstance(items, tuple):
            items = tuple([items])

        for k, item in enumerate(items):

            if not isinstance(item, (int, slice)):
                raise IndexError

            if isinstance(item, slice):

                if item.start == item.stop or item.start == self.shape[k]:
                    return MultiPoly.empty()
                
                index_coeffs = [ c[item] for c in self.coeffs ]
                return MultiPoly(self.kernel, index_coeffs)
            
            if isinstance(item, int):

                # index_coeffs = [ np.array([c[item]]) for c in self.coeffs ]
                index_coeffs = [ c[item] for c in self.coeffs ]

                return MultiPoly(self.kernel, index_coeffs)

    def __repr__(self):
        ''' Representation of MultiPoly '''

        if self.ndim == 0: type_text = "scalar"
        elif self.ndim == 1: type_text = "vector"
        elif self.ndim == 2: type_text = "matrix"

        poly_repr = f"{type_text.capitalize()} poly in x:\n"
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

    def _function(self, x):
        ''' Computes polynomial value at point x '''

        if isinstance(x, (list, np.ndarray)): 
            if len(x) != self._dim:
                raise TypeError("Input has incorrect dimensions.")
            s = np.zeros(self.shape)

        if isinstance(x, tuple):
            x = np.array(x)
            s = np.zeros(self.shape)

        if isinstance(x, MultiPoly):
            correct_dims = x.shape == (self.n,) or x.shape == (self.n,1) or x.shape == (1,self.n)
            if not correct_dims:
                raise TypeError("Input has incorrect dimensions.")
            s = MultiPoly.zeros(dim = self.n, shape = (self.shape))

        for mon, coeff in zip(self.kernel, self.coeffs):
            res_mon = np.prod([ x[i]**power for i, power in enumerate(mon) ])
            s += res_mon * coeff

        return s

    def _gradient(self, x):
        ''' Computes polynomial gradient at point x '''
        if self._poly_grad is None:
            self.poly_grad()
        return self._poly_grad(x)

    def _jacobian(self, x):
        ''' Computes polynomial Jacobian at point x '''
        if self._poly_jaco is None:
            self.poly_jaco()
        return self._poly_jaco(x)

    def _hessian(self, x):
        ''' Computes polynomial Hessian at point x '''
        if self._poly_hess is None:
            self.poly_hess()
        return self._poly_hess(x)

    def poly_diff(self):
        '''
        Computes the first-order polynomial derivatives with respect to each variable.
        OBS: can always be computed, no matter the polynomial shape. 
        To be used by gradient, jacobian and hessian methods.
        
        Returns self._poly_diff, a list with all self.n first order polynomial derivatives.
        '''
        self._poly_diff = []
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
            self._poly_diff.append( MultiPoly(kernel=diff_kernel, coeffs=diff_coeffs) )

        return self._poly_diff

    def poly_grad(self):
        ''' Compute gradient polynomial. For scalar polynomials only. '''

        if self.ndim != 0 and self.shape != (1,) and self.shape != ():
            raise NotImplementedError("Gradient is not implemented for non-scalar polynomials.")

        ''' If first-order derivatives are not computed yet, compute them. '''
        if self._poly_diff is None:
            self.poly_diff()

        kernel = list( set.union(*[ set(poly.kernel) for poly in self._poly_diff ]) )
        kernel_size = len(kernel)

        coeff_list = [ np.zeros(self.n) for _ in range(kernel_size) ]
        for i, poly in enumerate(self._poly_diff):
            for mon, coeff in zip(poly.kernel, poly.coeffs):
                id = kernel.index(mon)
                coeff_list[id][i] += coeff

        self._poly_grad = MultiPoly( kernel=kernel, coeffs=coeff_list )
        return self._poly_grad

    def poly_jaco(self):
        ''' Compute Jacobian polynomial. For vector polynomials only. '''

        if self.ndim != 1:
            raise NotImplementedError("Jacobian is not implemented for non-vector polynomials.")

        ''' If first-order derivatives are not computed yet, compute them. '''
        if self._poly_diff is None:
            self.poly_diff()

        kernel = list( set.union(*[ set(poly.kernel) for poly in self._poly_diff ]) )
        kernel_size = len(kernel)

        coeff_list = [ np.zeros(self.shape[0], self.n) for _ in range(kernel_size) ]
        for i, poly in enumerate(self._poly_diff):
            for mon, coeff in zip(poly.kernel, poly.coeffs):
                id = kernel.index(mon)
                coeff_list[id][:,i] += coeff

        self._poly_jaco = MultiPoly( kernel=kernel, coeffs=coeff_list )
        return self._poly_jaco

    def poly_hess(self):

        if self.ndim != 0 and self.shape != (1,) and self.shape != ():
            raise NotImplementedError("Hessian is not implemented for non-scalar polynomials.")

        if self._poly_grad is None:
            self.poly_grad()
        
        if self._poly_grad._poly_diff is None:
            self._poly_grad.poly_diff()

        diff2_polys = self._poly_grad._poly_diff

        kernel = list( set.union(*[ set(diff2_poly.kernel) for diff2_poly in diff2_polys ]) )
        kernel_size = len(kernel)

        coeff_list = [ np.zeros((self.n, self.n)) for _ in range(kernel_size) ]
        for i in range(self.n):
            for j, diff2_poly in enumerate(diff2_polys):
                for mon, coeff in zip(diff2_poly.kernel, diff2_poly.coeffs):
                    id = kernel.index(mon)
                    coeff_list[id][i,j] += coeff[i]

        self._poly_hess = MultiPoly( kernel=kernel, coeffs=coeff_list )
        return self._poly_hess

    def filter(self):
        '''
        Deletes entries of self.coeffs and self.kernel with zero coefficients.
        '''
        filtered_kernel, filtered_coeffs = [], []
        for mon, coef in zip(self.kernel, self.coeffs):

            to_delete = False
            to_delete = to_delete or isinstance(coef, sym.Expr) and coef == sym.S.Zero
            to_delete = to_delete or ( isinstance(coef, sym.MatrixExpr) and coef == sym.ZEROS(*coef.shape) )
            to_delete = to_delete or ( isinstance(coef, (int, float, np.ndarray)) and np.linalg.norm(coef) < 1e-12 )

            if not to_delete:
                filtered_kernel.append(mon)
                filtered_coeffs.append(coef)

        self.kernel = filtered_kernel
        self.coeffs = filtered_coeffs

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

    def outer(poly1, poly2):
        ''' Outer product between polynomials '''

        if not isinstance( poly1, MultiPoly ) or not isinstance( poly2, MultiPoly ):
            raise TypeError("Polynomials must be instances of the MultiPoly class")
        
        if poly1.ndim != 1 or poly2.ndim != 1:
            raise TypeError("Polynomials must be vectors.")
        
        return MultiPoly._multiply(poly1, poly2, op = 0)

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

    def compute_sos_kernel(self):
        '''
        Function for computing the minimal polynomial kernel for SOS decomposition. 
        '''
        # start with an empty SOS kernel
        self.sos_kernel = []

        # loop through each monomial in the polynomial kernel
        for mon in self.kernel:

            # Compute the set of all possible sums of 2 monomials currently in the SOS kernel
            poss_sums = set([ tuple(np.array(mon1) + np.array(mon2)) for mon1, mon2 in itertools.combinations(self.sos_kernel, 2) ])

            # If the current monomial can be produced by at least one of the possible sums from above, jump to next iteration
            if mon in poss_sums: 
                continue

            # if the SOS kernel has only one monomial, add the current monomial to the SOS kernel
            if len(poss_sums) == 0:
                self.sos_kernel.append(mon)
                continue

            ''' 
            If the current monomial cannot be formed from possible sums on the current SOS kernel, 
            check if its possible to create it from them...
            '''
            possibilities = []

            # If all exponents of mon are even, it can be created from 
            if np.all([ exp % 2 == 0 for exp in mon ]):
                possibilities.append( tuple([int(exp/2) for exp in mon]) )

            # Checks if mon can be created from the combination of monomials already in self._sos_monomials and another
            for sos_mon in self.sos_kernel:
                pos = np.array(mon) - np.array(sos_mon)
                if np.all(pos >= 0): 
                    possibilities.append( tuple([ int(exp) for exp in pos ]) )

            index = np.argmin([ np.linalg.norm(pos) for pos in possibilities ])
            new_sos_mon = possibilities[index]
            if new_sos_mon not in self.sos_kernel:
                self.sos_kernel.append(new_sos_mon)

    def compute_sos_indexes(self):
        '''
        Computes the index matrix representing the rule for placing the coefficients in the correct places on the 
        shape matrix of the SOS representation. Algorithm gives preference for putting the elements of coeffs 
        closer to the main diagonal of the SOS matrix.
        '''
        # Compute SOS kernel first
        if self.sos_kernel is None: 
            self.compute_sos_kernel()
        sos_dim = len(self.sos_kernel)

        self.sos_locations = [ [] for _ in self.kernel ]
        self.sos_indexes = -np.ones([sos_dim, sos_dim], dtype='int')
        for k in range(len(self.kernel)):

            mon = self.kernel[k]

            # Checks the possible (i,j) locations on SOS matrix where the monomial can be put
            kth_poss_places = []
            for (i,j) in itertools.product(range(sos_dim),range(sos_dim)):
                if i > j: continue
                sos_mon_i, sos_mon_j = np.array(self.sos_kernel[i]), np.array(self.sos_kernel[j])

                if mon == tuple(sum([sos_mon_i, sos_mon_j])):
                    kth_poss_places.append( (i,j) )

            # From these, chooses the place closest to SOS matrix diagonal
            distances_from_diag = np.array([ np.abs(place[0] - place[1]) for place in kth_poss_places ])
            i,j = kth_poss_places[np.argmin(distances_from_diag)]

            self.sos_locations[k] = kth_poss_places
            self.sos_indexes[i,j] = k

    def shape_matrix(self):
        '''
        Using the SOS kernel and index matrix, returns the SOS shape matrix correctly populated by the coefficients.
        '''
        if self.ndim != 0:
            raise NotImplementedError("Currently, the shape matrix can only be compute for scalar polynomials.")
        
        if len(self.coeffs) != len(self.kernel):
            raise Exception("The number of coefficients must be equal to the kernel dimension.")
        
        if self.sos_indexes is None:
            self.compute_sos_indexes()

        sos_kernel_dim = len(self.sos_kernel)
        sos = np.zeros([sos_kernel_dim, sos_kernel_dim])

        for (i,j) in itertools.product(range(sos_kernel_dim),range(sos_kernel_dim)):

            if i > j: 
                continue

            k = self.sos_indexes[i,j]
            if k >= 0:
                if i == j:
                    sos[i,j] = self.coeffs[k]
                else:
                    sos[i,j] = 0.5 * self.coeffs[k]
                    sos[j,i] = 0.5 * self.coeffs[k]
        return sos

    def sos_decomposition(self, verb=False):
        '''
        Returns a suitable SOS decomposition, if the polynomial is SOS. 
        Only works for scalar polynomials.
        '''
        if self.sos_locations is None:
            self.compute_sos_indexes()

        sos_dim = len(self.sos_kernel)
        SOS = cvx.Variable( (sos_dim, sos_dim), symmetric=True )
        SOSt = cvx.Variable( (sos_dim, sos_dim), symmetric=True )
        constraints = [ SOS >> 0 ]

        norm_const = min(np.abs(self.coeffs))
        if norm_const < 1e-3: norm_const = 1.0

        for locs, c in zip(self.sos_locations, self.coeffs):
            constraints += [ sum([ SOSt[index] if index[0]==index[1] else 2*SOSt[index] for index in locs ]) == c/norm_const ]

        prob = cvx.Problem( cvx.Minimize( cvx.norm( SOS - SOSt ) ), constraints )
        prob.solve(solver = 'SCS', verbose=False)

        if verb: print(f"SOS decomposition problem returned with status {prob.status}")

        if "optimal" in prob.status:
            # If SOS, multiply resulting matrix by normalization coefficient
            self.sos_matrix = norm_const * SOSt.value
        else:
            self.sos_matrix = self.shape_matrix()

        return self.sos_matrix

    def validate_sos(self):
        '''
        Debug method for SOS validation.
        '''
        if self.sos_matrix is None: 
            self.sos_decomposition()

        errors = []
        for locs, c in zip(self.sos_locations, self.coeffs):
            errors += [ sum([ self.sos_matrix[index] if index[0]==index[1] else 2*self.sos_matrix[index] for index in locs ]) - c ]

        return sum(np.abs(errors))

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

    def frame_transform(self, translation: np.ndarray, rotation: np.ndarray):
        '''
        Performs a rigid-body transformation x = t + R(theta) x' to the polynomial variables.
        Returns a new polynomial in the coordinates of the new frame.
        '''
        if isinstance(translation, (list, np.ndarray, tuple)) and len(translation) != self.n:
            raise Exception("Length has incorrect dimensions.")
        
        if not isinstance(rotation, np.ndarray) or rotation.shape != (self.n, self.n) or np.linalg.det(rotation) - 1.0 > 1e-3:
            raise Exception("Rotation must be an SO(n) matrix.")

        EYE = np.eye(self.n, dtype='int')
        x_newframe = MultiPoly(kernel=[ tuple(EYE[k,:]) for k in range(self.n) ], 
                               coeffs=[ np.array(EYE[k,:], dtype='float') for k in range(self.n) ])
        
        translation = np.array(translation)
        rotation_poly = MultiPoly(kernel=[(0,0)], coeffs=[rotation])
        x = rotation_poly @ x_newframe - translation
        
        return self(x)

    def inverse_gamma_transform(self, x: np.ndarray, gamma: Poly):
        '''
        Assuming the MultiPoly was obtained by a gamma transformation, 
        get the inverse transformation, gradient and Hessian, computed at point x.
        
        Input:    - point array
                  - K infinity gamma polynomial
        Returns:  (i) V(x), (ii) ∇V(x), (iii) Hv(x)
        '''
        if not isinstance(gamma, Poly):
            raise TypeError("Gamma function is not a polynomial.")
        
        gamma_der = Poly( polyder(gamma.coef) )
        gamma_int = Poly( polyint(gamma.coef) )
        
        ''' Compute V(x) '''
        barV = self(x)

        V0 = 100*np.random.rand()
        sol = least_squares( lambda V: barV - gamma_int( V ), V0 )
        V = sol.x

        if sol.fun >= 1e-10:
            raise ValueError("Optimization did not converge.")

        ''' Compute ∇V(x) '''
        if self._poly_grad is None:
            self.poly_grad()
        
        gammaV = gamma(V)
        bar_gradV = self._poly_grad(x)
        gradV = bar_gradV / gammaV

        ''' Compute Hv(x) '''
        if self._poly_hess is None:
            self.poly_hess()

        gammaderV = gamma_der(V)
        bar_Hv = self._poly_hess(x)
        Hv = ( 1/gammaV ) * ( bar_Hv - gammaderV * np.outer( gradV,gradV ) )

        return V, gradV, Hv

    def save(self, filename):
        ''' Saves object into a file '''

        multipoly_log = {"kernel": self.kernel,
                         "coeffs": [ coef.tolist() if isinstance(coef, np.ndarray) else coef for coef in self.coeffs ],
                         "sos_kernel": self.sos_kernel,
                         "sos_indexes": None }
        
        if self.sos_indexes is not None:
            multipoly_log["sos_indexes"] = self.sos_indexes.tolist()

        try:
            with open("polynomials/" + filename + ".json", "w") as file:
                print("Saving MultiPoly data...")
                json.dump(multipoly_log, file, indent=0)
        except IOError:
            print("Couldn't save polynomial." + IOError)

    def kernel_value(mons: list, input: float | list) -> np.ndarray:
        ''' 
        Compute the kernel value for a given input.
        Mons is an array of monomials and input is the corresponding input to be computed. 
        Returns: array with monomial values
        '''
        if isinstance( input, (np.float64, np.int64, int, float)):
            input = [input]

        kern_arr = np.zeros(len(mons))
        for k, mon in enumerate(mons):

            if len(input) != len(mon):
                raise TypeError("Input dimensions are wrong.")
            
            kern_arr[k] = 1.0
            for i, power in enumerate(mon):
                kern_arr[k] *= input[i]**power

        return kern_arr

    @property
    def T(self):
        ''' Transpose operation, defined for array-like coefficients '''

        transposed_coeffs = self.coeffs
        if self.ndim == 2:
            transposed_coeffs = [ coeff.T for coeff in self.coeffs ]            

        return MultiPoly( kernel=self.kernel, coeffs=transposed_coeffs )

    @classmethod
    def load(cls, filename):
        ''' Loads a multipoly file '''

        try:
            with open("polynomials/" + filename + ".json") as file:
                print("Loading shape file: " + filename + ".json")
                multipoly_log = json.load(file)

                kernel = [ tuple(mon) for mon in multipoly_log["kernel"] ]
                coeffs = [ np.array(coef) for coef in multipoly_log["coeffs"] ]
                poly = cls(kernel=kernel, coeffs=coeffs)

                if multipoly_log["sos_kernel"] != None:
                    poly.SOSkernel = [ tuple(mon) for mon in multipoly_log["sos_kernel"] ]

                if multipoly_log["sos_indexes"] != None:
                    poly.SOSindexes = np.array(multipoly_log["sos_indexes"])

                return poly

        except IOError:
            print("Couldn't locate polynomial file" + filename + ".json")

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
    
    @classmethod
    def from_nppoly(cls, poly: np.polynomial.Polynomial):
        ''' Converts a numpy polynomial into a Multipoly '''

        kernel, coeffs = [], []
        for k, c in enumerate(poly.coef):
            kernel.append( tuple([k]) )
            coeffs.append(c)

        return cls(kernel, coeffs, dim=1)