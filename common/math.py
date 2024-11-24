import numpy as np
import scipy as sp
import operator

from numpy.random import rand, randn, randint
from numpy.polynomial import Polynomial as Poly
from dataclasses import dataclass
from itertools import product

@dataclass
class Eigen():
    ''' 
    Data class for generalized eigenvalues/eigenvectors.
    ( βeta M - alpha N ) r_eigenvectors = 0 or
    l_eigenvectors' ( βeta M - alpha N ) = 0
    '''
    alpha: complex
    beta: float
    eigenvalue: complex
    rightEigenvectors: list | np.ndarray
    leftEigenvectors: list | np.ndarray
    inertia: float                  # value of z_left' M z_right

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
        coef can be:

        '''
        if isinstance(coef, list):
            self.num_coef = len(coef)
            self.shape = coef[0].shape
        elif isinstance(coef, dict):
            powers = coef.keys()
            self.num_coef = len(coef)
            self.shape = coef[powers[0]].shape
        elif isinstance(coef, np.ndarray) and coef.ndim == 3:
            self.num_coef = coef.shape[0]
            self.shape = (coef.shape[1], coef.shape[2])
        else:
            raise TypeError("MatrixPolynomial must receive a list/dict of coefficients.")

        if self.shape == (self.shape[0],):
            self.shape = (self.shape[0],1)

        if self.shape[0] == self.shape[1]:
            self.type = 'regular'
        else: 
            self.type = 'singular'

        # Initializes coefficients and polynomial array
        self.elements = np.zeros((self.num_coef, self.shape[0], self.shape[1]))
        self.coef = [ np.zeros(self.shape) for _ in range(self.num_coef) ]
        self.poly_array: np.ndarray[Poly] = np.array([[ Poly([0.0]) for _ in range(self.shape[1]) ] for _ in range(self.shape[0]) ])

        # Sets coefficients
        self.update(coef)

    def __call__(self, l):
        '''
        MatrixPolynomial call method.
        Returns: - np.ndarray value of P(λ) = P₀ + λ P₁ + λ² P₂ + ... for given λ.
        '''
        return np.sum([ (l**k) * c for k, c in enumerate(self.coef) ])

    def __repr__(self) -> str:
        '''
        Representation method for the matrix polynomial P(λ) = P₀ + λ P₁ + λ² P₂ + ... 
        '''
        np.set_printoptions(precision=3, suppress=True)
        ret_str = f'({self.shape[0]} x {self.shape[1]}) ' + '{}'.format(type(self).__name__) + f" on {self.symbol}: P({self.symbol}) = P₀"
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

        if isinstance(op1, MatrixPolynomial): op1 = op1.poly_array
        if isinstance(op2, MatrixPolynomial): op2 = op2.poly_array

        return op2 + type * op2

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
        if isinstance(op1, MatrixPolynomial): op1 = op1.poly_array
        if isinstance(op2, MatrixPolynomial): op2 = op2.poly_array

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

    def _update_poly_array(self):
        '''
        Updates ndarray of numpy Polynomials using the elements.
        '''
        for (power, i, j), ele in np.ndenumerate(self.elements):

            curr_poly = self.poly_array[i,j]
            num_powers_toadd = power - curr_poly.degree()
            if num_powers_toadd > 0:
                self.poly_array[i,j].coef = np.hstack([ curr_poly.coef, [0.0 for _ in range(num_powers_toadd) ] ])

            self.poly_array[i,j].coef[power] = ele

    def update(self, coef: list | dict):
        '''
        Update method for matrix polynomial. Existing coefficients can be modified, 
        but the polynomial shape cannot be changed after creation.

        Input: - list [ coef0, coef1, ... , coefN ] of ordered powers OR
               - dict { 0: coef0, 1: coef1, ... , N: coefN } of (power:coef) pairs 
                        (repeating power keys are not allowed and return an error)

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

            if coef.ndim <= 2 and coef[*[ randint(coef.shape[n]) for n in range(coef.ndim) ]] :
                pass

        self.coef = [ c for c in self.elements ]
        self._update_poly_array()

        ''' Updates the number of coefs, max degree and generalized eigenvalues '''
        self.num_coef = self.elements.shape[0]
        self.degree = self.num_coef - 1

    @property
    def T(self):
        ''' Transpose operation '''
        return self.poly_array.T

    @classmethod
    def from_array(cls, poly_array: np.ndarray[Poly]):
        '''
        Creates a MatrixPolynomial object using an ndarray of (scalar) Polynomials
        '''
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

        return cls(coefs)