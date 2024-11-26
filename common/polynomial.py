import operator
import warnings

import numpy as np
import scipy as sp

from numpy.polynomial import Polynomial as Poly

from .pencil import MatrixPencil

class MatrixPolynomial():
    '''
    Class for matrix polynomials of the form P(λ) = P₀ + λ P₁ + λ² P₂ + ... of finite degree.
    '''
    SUBS_INTS = '₀₁₂₃₄₅₆₇₈₉'
    SUPS_INTS = '⁰¹²³⁴⁵⁶⁷⁸⁹'

    def __init__(self, coef, **kwargs):
        
        self.symbol = 'λ'
        for key, item in kwargs.items():
            if key.lower() == 'symbol':
                if isinstance(item, str):
                    self.symbol = item
                    continue
                raise TypeError("Polynomial variable name must be a string.")

        '''
        coef can be : - list [ coef0, coef1, ... , coefN ] of ordered powers OR
                      - dict { 0: coef0, 1: coef1, ... , N: coefN } of (power:coef) pairs 
                        (repeating power keys are not allowed and return an error)
                      - np.ndarray with ndim = 3 with each element being (powers, i, j)
        '''
        if isinstance(coef, list):
            self.num_coef = len(coef)
            self.ndim = coef[0].ndim
            self.shape = coef[0].shape
        elif isinstance(coef, dict):
            powers = coef.keys()
            self.num_coef = len(coef)
            self.ndim = coef[powers[0]].ndim
            self.shape = coef[powers[0]].shape
        elif isinstance(coef, np.ndarray) and coef.ndim == 3:
            self.num_coef = coef.shape[0]
            self.ndim = coef.ndim-1
            self.shape = coef[0].shape
        else:
            raise TypeError("MatrixPolynomial must receive a list/dict of coefficients.")

        self.vector_like = self.ndim == 1
        self.matrix_like = self.ndim == 2
        self.is_square = self.matrix_like and (self.shape[0] == self.shape[1])

        if self.is_square: self.type = 'regular'
        else: self.type = 'singular'

        # Initializes coefficients and polynomial array
        shape_of_elements = tuple([self.num_coef] + [ self.shape[dim] for dim in range(self.ndim) ])
        self.elements = np.zeros(shape_of_elements)         # 3D array with (powers, i, j)
        self.coef = [ np.zeros(self.shape) for _ in range(self.num_coef) ]

        self.poly_array: np.ndarray[Poly] = np.zeros(self.shape, dtype=Poly)
        for index, _ in np.ndenumerate(self.poly_array):
            self.poly_array[index] = Poly([0.0], symbol=self.symbol)

        # Sets coefficients
        self.update(coef=coef)

        ''' Parameters '''
        self.realEigenTol = 1e-10           # Tolerance to consider an eigenvalue as real
        self.max_order = 20                 # Max. polynomial order to compute nullspace solutions

    def __power__(self, power):
        ''' Matrix power '''
        return np.linalg.matrix_power(self.poly_array, power)

    def __pos__(self):
        return self.poly_array

    def __neg__(self):
        return -self.poly_array

    def __getitem__(self, items):
        ''' Subscritable method '''
        return self.poly_array[items]

    def __call__(self, l):
        '''
        MatrixPolynomial call method.
        Returns: - np.ndarray value of P(λ) = P₀ + λ P₁ + λ² P₂ + ... for given λ.
        '''
        return sum([ (l**k) * c for k, c in enumerate(self.coef) ])

    def __repr__(self) -> str:
        '''
        Representation method for the matrix polynomial P(λ) = P₀ + λ P₁ + λ² P₂ + ... 
        '''
        np.set_printoptions(precision=3, suppress=True)
        ret_str = f"{self.shape[0]}"
        ret_str += "".join([ f" x {self.shape[dim]}" for dim in range(1,self.ndim) ])
        ret_str += "".join([ ' {}'.format(type(self).__name__), f" on {self.symbol}: P({self.symbol}) = P₀"])
        for k in range(1, self.degree+1):
            k_str = str(k)
            power_str = "".join([ MatrixPolynomial.SUPS_INTS[int(k_str[i])] for i in range(len(k_str)) ])
            index_str = "".join([ MatrixPolynomial.SUBS_INTS[int(k_str[i])] for i in range(len(k_str)) ])
            ret_str += f' + {self.symbol}' + power_str + ' P' + index_str

        return ret_str

    def __str__(self) -> str:
        '''
        Print the matrix polynomial P(λ) = P₀ + λ P₁ + λ² P₂ + ... 
        '''
        ret_str = self.__repr__() + ' with\n'
        for k in range(0, self.degree+1):
            k_str = str(k)
            index_str = "".join([ MatrixPolynomial.SUBS_INTS[int(k_str[i])] for i in range(len(k_str)) ])
            ret_str += f'P{index_str} = \n'
            ret_str += f'{self.coef[k]}'
            if k < self.degree: ret_str += '\n'

        return ret_str

    def _verify_coef(self, coef):
        '''
        Verify if a passed coefficient is valid.
        '''
        if not isinstance(coef, np.ndarray) or coef.shape != self.shape:
            error_msg = f"Passed coefficient of type {type(coef)}"
            if isinstance(coef, np.ndarray):
                error_msg += f" of shape {coef.shape}"
            error_msg += f" is not compatible with {np.ndarray} of shape {self.shape}."
            raise TypeError(error_msg)
        
    def _add_sub(op1, op2, type):
        '''
        MatrixPolynomial addition/subtraction
        '''
        if np.abs(type) != 1:
            raise ValueError("\"op\" argument should be +1 or -1.")

        if isinstance(op1, MatrixPolynomial): 
            op1 = op1.poly_array
        if isinstance(op2, MatrixPolynomial): 
            op2 = op2.poly_array

        return op1 + type * op2

    def _add(op1, op2):
        return MatrixPolynomial._add_sub(op1, op2, type=+1)

    def _sub(op1, op2):
        return MatrixPolynomial._add_sub(op1, op2, type=-1)

    def _multiply(op1, op2, type):
        ''' 
        MatrixPolynomial multiplication:      type = +1 for term-by-term, 
                                              type = -1 for matrix-like,
                                              type = 0 for outer product
        '''
        if isinstance(op1, MatrixPolynomial): 
            op1 = op1.poly_array
        if isinstance(op2, MatrixPolynomial): 
            op2 = op2.poly_array

        if type == 1:
            return op1 * op2
        if type == -1:
            return op1 @ op2
        if type == 0:
            return np.outer( op1, op2 )

    def _mul(op1, op2):
        return MatrixPolynomial._multiply(op1, op2, type=+1)

    def _matmul(op1, op2):
        return MatrixPolynomial._multiply(op1, op2, type=-1)

    def _operator_fallbacks(operation, op_name):
        '''
        Implementation of forward, reverse and inplace operations for MatrixPolynomial.
        '''
        def forward(op1, op2): 
            return operation(op1, op2)
        
        def reverse(op1, op2): 
            return operation(op1, op2)
        
        def inplace(op1, op2): 
            return operation(op1, op2)

        forward.__name__ = '__' + op_name.__name__ + '__'
        forward.__doc__ = operation.__doc__

        reverse.__name__ = '__r' + op_name.__name__ + '__'
        reverse.__doc__ = operation.__doc__

        inplace.__name__ = '__i' + op_name.__name__ + '__'
        inplace.__doc__ = operation.__doc__

        return forward, reverse, inplace

    __add__, __radd__, __iadd__ = _operator_fallbacks(_add, operator.add)
    __sub__, __rsub__, __isub__ = _operator_fallbacks(_sub, operator.sub)
    __mul__, __rmul__, __imul__ = _operator_fallbacks(_mul, operator.mul)
    __matmul__, __rmatmul__, __imatmul__ = _operator_fallbacks(_matmul, operator.matmul)

    def _update_poly_array(self):
        '''
        Updates ndarray of numpy Polynomials using the elements.
        '''
        for index_ele, ele in np.ndenumerate(self.elements):
            power = index_ele[0]
            index = index_ele[1:]
            curr_poly = self.poly_array[index]
            num_powers_toadd = power - curr_poly.degree()
            if num_powers_toadd > 0:
                self.poly_array[index].coef = np.hstack([ curr_poly.coef, [0.0 for _ in range(num_powers_toadd) ] ])
            self.poly_array[index].coef[power] = ele

    def _symm(self, type = +1):
        '''
        Returns equivalent symmetric/antisymmetric matrix polynomial (or linear matrix pencil).
        '''
        if not self.is_square:
            if type == +1: type_text = "symmetric"
            if type == -1: type_text = "antisymmetric"
            msg_txt = f"Cannot compute {type_text} part of non-square matrix polynomial."
            raise Exception(msg_txt)
        
        if not isinstance(self, MatrixPencil):
            return MatrixPolynomial( coef=[ 0.5*(c + type * c.T) for c in self.coef ] )
        else:
            return MatrixPencil( M = 0.5*(self.M + type * self.M.T), N = 0.5*(self.N + type * self.N.T) )

    def outer(poly1, poly2):
        ''' Outer product between MatrixPolynomials '''
        return MatrixPolynomial._multiply(poly1, poly2, type = 0)

    def update(self, coef: list | dict):
        '''
        Update method for matrix polynomial. Existing coefficients can be modified, 
        but the polynomial shape cannot be changed after creation.

        Input: - list [ coef0, coef1, ... , coefN ] of ordered powers OR
               - dict { 0: coef0, 1: coef1, ... , N: coefN } of (power:coef) pairs 
                        (repeating power keys are not allowed and return an error)
               - np.ndarray with ndim = 3 with each element being (powers, i, j)

        OBS: this setting method is APPENDING, meaning that 
        if the number of passed coefficients exceeds the current 
        number of coefficients, it appends new coefficients.
        '''
    
        ''' (Default) - updates by ordered sequence of increasing powers (numpy Polynomial style) '''
        if isinstance(coef, list):
            for k, c in enumerate(coef):

                # verify is coefficient is valid
                self._verify_coef(c)

                # appends new coefficient if necessary
                if k >= self.num_coef:
                    self.elements = np.append(self.elements, c[np.newaxis,], axis=0)

                # updates coefficient
                self.elements[k] = c                                                    

        ''' Updates by dictionary of (power:coefficient) '''
        if isinstance(coef, dict):
            powers = []
            for k, c in coef.items():
                
                # Verify power entries
                if ( isinstance(k, int) and k >= 0 ) or ( isinstance(k, str) and str.isnumeric(k) and int(k) >= 0 ):
                    k = int(k)
                    if k not in powers: 
                        powers.append(k)
                    else:
                        raise TypeError("Duplicate entry of same power.")
                else:
                    raise TypeError("Dictionary keys must represent polynomial powers.")
                
                # verify is coefficient is valid
                self._verify_coef(c)

                # appends new coefficients if necessary
                if k >= len(self.coef):
                    num_extra_coefs = k - len(self.coef) + 1
                    extra_zeros = np.zeros((num_extra_coefs, self.shape[0], self.shape[1]))
                    self.elements = np.append(self.elements, extra_zeros, axis=0)

                # updates coefficient
                self.elements[k] = c

        ''' Updates by np.ndarray os elements of signature (powers, i, j) '''
        if isinstance(coef, np.ndarray):

            if coef.ndim == 3:
                self.elements = coef

            if coef.ndim <= 2 and coef[*[ np.random.randint(coef.shape[n]) for n in range(coef.ndim) ]] :
                pass

        self.coef = [ c for c in self.elements ]
        self._update_poly_array()

        ''' Updates the number of coefs, max degree and generalized eigenvalues '''
        self.num_coef = self.elements.shape[0]
        self.degree = self.num_coef - 1

    def nullspace(self):
        ''' 
        Returns the minimum (right) nullspace polynomial of P(λ), that is,
        a new polynomial matrix N(λ) = N0 + λ N1 + λ² N2 + ... of P(λ) satisfying P(λ) N(λ) = 0.

        Inputs: - max_degree is the maximum possible degree of N(λ)
        '''
        if not self.is_square:
            raise Exception("Vmatrix does not have a non-trivial right nullspace.")
        
        n, m = self.shape[0], self.shape[1]
        for Qdegree in range(0, self.max_order+1):

            Vmatrix = np.zeros([ n*(self.degree + Qdegree + 1), m*(Qdegree+1) ])
            
            sliding_list = [ c for c in self.coef ]
            zeros = [ np.zeros((n,m)) for _ in range(Qdegree) ]
            sliding_list = zeros + sliding_list + zeros
            
            for i in range(self.degree + Qdegree + 1):
                l = sliding_list[i:i+Qdegree+1]
                l.reverse()
                Vmatrix[i*n:(i+1)*n,:] = np.hstack(l)

            Null = sp.linalg.null_space(Vmatrix)
            if Null.size != 0:
                break

            if Qdegree == self.max_order:
                warnings.warn("P(λ) likely does not have a non-trivial right nullspace.")
                return None

        Ncoefs = [ Null[i*m:(i+1)*m,:] for i in range(Qdegree+1) ]
        return MatrixPolynomial(coef=Ncoefs, symbol=self.symbol)

    def symmetric(self):
        '''
        Returns equivalent symmetric matrix polynomial.
        '''
        self._symm(type=+1)

    def antisymmetric(self):
        '''
        Returns equivalent antisymmetric matrix polynomial.
        '''
        self._symm(type=-1)

    def companion_form(self):
        ''' 
        Returns a MatrixPencil that is the equivalent companion form of P(λ).
        '''
        if self.shape[0] != self.shape[1]:
            raise NotImplementedError("Companion form is only implemented for square matrix polynomials. ")
        
        n = self.shape[0]
        deg = self.degree

        # Builds block diagonal matrix M
        M = np.zeros((deg*n,deg*n))
        M = sp.linalg.block_diag(*([ self.coef[-1] ] + [ np.eye(n) for _ in range(deg-1) ]))

        # Builds row companion form matrix N
        N = np.zeros((deg*n,deg*n))
        for i in range(deg):
            N[0:n, i*n:(i+1)*n] = -self.coef[deg-1-i]                 # First row block
            if i < deg-1:
                N[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = np.eye(n)    # Additional -I's at the diagonal below

        return MatrixPencil( M, N )

    @property
    def T(self):
        ''' Transpose operation '''
        return MatrixPolynomial.from_array( self.poly_array.T )

    @classmethod
    def from_array(cls, poly_array: np.ndarray[Poly]):
        '''
        Creates a MatrixPolynomial object using an ndarray of (scalar) Polynomials
        '''
        if isinstance(poly_array, np.polynomial.Polynomial):
            return poly_array

        if not isinstance(poly_array, np.ndarray):
            raise TypeError("Polynomial must be an ndarray.")

        coefs = []
        for index, poly in np.ndenumerate(poly_array):

            if not isinstance(poly, Poly):
                raise TypeError("Array elements must be numpy Polynomials.")
            
            # Creates new coefficients as necessary
            num_coefs_toadd = len(poly.coef) - len(coefs)
            if num_coefs_toadd > 0:
                for _ in range(num_coefs_toadd):
                    coefs.append( np.zeros(poly_array.shape) )

            # Populate given coefficient
            for k, c in enumerate(poly.coef): 
                coefs[k][index] = c

        return cls(coefs, symbol=poly.symbol)