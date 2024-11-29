import math
import operator
import warnings

import numpy as np
import scipy as sp
import cvxpy as cvx

from itertools import product
from dataclasses import dataclass
from scipy.linalg import null_space
from numpy.polynomial import Polynomial as Poly

def to_coef(poly_arr: np.ndarray[Poly]) -> tuple[ list[np.ndarray], str ]:
    '''
    Input:   - np.ndarray of numpy Polynomials or
    Returns: - list of np.ndarray coefficients.
    '''
    if not isinstance(poly_arr, np.ndarray):
        raise TypeError("Input must be an np.ndarray.")

    # get the symbol
    symbol = 'x'
    for index, p in np.ndenumerate(poly_arr):
        if hasattr(p, "symbol"): 
            symbol = p.symbol
            break

    # populate coefs
    coefs = []
    for index, p in np.ndenumerate(poly_arr):

        if isinstance(p, (float, np.float32, int, np.int32)):
            p = Poly([p], symbol=symbol)
        elif not isinstance(p, Poly):
            raise TypeError("Input must by an np.ndarray of numpy Polynomials.")
        else:
            if p.symbol != symbol:
                raise Exception("Polynomials do not have the same symbol.")

        # Creates new coefficients as necessary
        if len(p.coef) > len(coefs):
            for _ in range(len(p.coef) - len(coefs)):
                coefs.append( np.zeros(poly_arr.shape) )

        # Populate given coefficient
        for k, c in enumerate(p.coef):
            coefs[k][index] = c

    return coefs, symbol

def to_array(coef: list[np.ndarray], symbol='x') -> np.ndarray[Poly]:
    '''
    Input:   - list of np.ndarray coefficients
    Returns: - np.ndarray of numpy Polynomials
    '''
    shape = coef[0].shape
    poly_arr = np.zeros(shape, dtype=Poly)
    for index, _ in np.ndenumerate(poly_arr):
        poly_arr[index] = Poly([ c[index] for c in coef ], symbol=symbol)

    return poly_arr

def nullspace(poly_arr: np.ndarray[Poly], max_order = 20) -> np.ndarray[Poly]:
    ''' 
    Returns the minimum (right) nullspace polynomial of P(λ), that is,
    a new polynomial matrix N(λ) = N0 + λ N1 + λ² N2 + ... of P(λ) satisfying P(λ) N(λ) = 0.

    Inputs: - np.ndarray of numpy Polynomials
            - max_degree is the maximum possible degree of N(λ)
    '''
    if not isinstance(poly_arr, np.ndarray):
        raise TypeError("Input must be a numpy array of Polynomials.")

    if poly_arr.ndim == 1:
        poly_arr = poly_arr.reshape(1,-1)

    coefs, symbol = to_coef(poly_arr)
    degree = len(coefs)-1

    n, m = poly_arr.shape[0], poly_arr.shape[1]
    for Qdegree in range(0, max_order+1):

        Vmatrix = np.zeros([ n*(degree + Qdegree + 1), m*(Qdegree+1) ])
        
        sliding_list = [ c for c in coefs ]
        zeros = [ np.zeros((n,m)) for _ in range(Qdegree) ]
        sliding_list = zeros + sliding_list + zeros
        
        for i in range(degree + Qdegree + 1):
            l = sliding_list[i:i+Qdegree+1]
            l.reverse()
            Vmatrix[i*n:(i+1)*n,:] = np.hstack(l)

        Null = sp.linalg.null_space(Vmatrix)
        if Null.size != 0:
            break

        if Qdegree == max_order:
            warnings.warn("P(λ) likely does not have a non-trivial right nullspace.")
            return None

    Ncoefs = [ Null[i*m:(i+1)*m,:] for i in range(Qdegree+1) ]
    poly_array = to_array(coef=Ncoefs, symbol=symbol)

    return poly_array

def companion_form(matrix_poly: list[np.ndarray] | np.ndarray[Poly]):
    ''' 
    Returns the M and N (deg*n, deg*n)-matrices of a linear matrix pencil λ M - N that is the equivalent companion form of a given matrix polynomial.
    '''
    if isinstance(matrix_poly, np.ndarray):
        coefs = to_coef( matrix_poly )
    elif isinstance(matrix_poly, list):
        coefs = matrix_poly
    else:
        raise TypeError("Input should be a list of coefficients or np.ndarray of numpy Polynomials.")

    ndim = coefs[0].ndim
    shape = coefs[0].shape

    if ndim != 2 and shape[0] != shape[1]:
        raise NotImplementedError("Companion form is only implemented for square matrix polynomials. ")
    
    n = shape[0]
    deg = len(coefs) - 1

    # Builds block diagonal matrix M
    M = np.zeros((deg*n,deg*n))
    M = sp.linalg.block_diag(*([ coefs[-1] ] + [ np.eye(n) for _ in range(deg-1) ]))

    # Builds row companion form matrix N
    N = np.zeros((deg*n,deg*n))
    for i in range(deg):
        N[0:n, i*n:(i+1)*n] = -coefs[deg-1-i]                 # First row block
        if i < deg-1:
            N[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = np.eye(n)    # Additional -I's at the diagonal below

    return M, N

def solve_poly_linearsys(T: np.ndarray, S: np.ndarray, b_poly: np.ndarray) -> np.ndarray:
    '''
    Finds the polynomial array x(λ) that solves (λ T - S) x(λ) = b(λ), where T, S are n x n (n=1 or n=2)
    and b(λ) is a polynomial array of size nxr or nxr.

    Input: - matrices T, S from linear matrix pencil (λ T - S)
    '''
    if isinstance(T, (int, float)): T = np.array([[ T ]])
    if isinstance(S, (int, float)): S = np.array([[ S ]])

    if T.shape != S.shape:
        raise TypeError("T and S must have the same shape.")
    
    if T.shape[0] != T.shape[1]:
        raise TypeError("T and S must be square matrices.")

    n = T.shape[1]
    r = b_poly.shape[1]

    if n != b_poly.shape[0]:
        raise TypeError("Number of lines in (λ T - S) and b(λ) must be the same.")

    # Extract max. degree of b_poly
    max_deg = 0
    for (i,j), poly in np.ndenumerate(b_poly):
        if not isinstance( poly, Poly ):
            raise TypeError("b(λ) is not an array of polynomials.")
        max_deg = max( max_deg, poly.degree() )

    # Initialize and populate bsys
    bsys = np.zeros(((max_deg+1)*n,r))
    for (i,j), poly in np.ndenumerate(b_poly):
        for k, c in enumerate(poly.coef):
            bsys[ k * n + i, j ] = c

    #  Initialize and populate Asys
    Asys = np.zeros(((max_deg+1)*n, max_deg*n))
    for i in range(max_deg):
        Asys[ i*n:(i+1)*n , i*n:(i+1)*n ] = -S
        Asys[ (i+1)*n:(i+2)*n , i*n:(i+1)*n ] = T

    results = np.linalg.lstsq(Asys, bsys, rcond=None)
    x_coefs = results[0]
    res = results[1]
    residue = np.linalg.norm(res)

    residue_tol = 1e-11
    if residue > residue_tol:
        warnings.warn(f"Large residue detected on linear system solution = {residue}")

    if max_deg == 0: max_deg = 1
    x_poly = np.array([[ Poly([0.0 for _ in range(max_deg) ], symbol='λ') for j in range(r) ] for i in range(n) ])
    for (i,j), c in np.ndenumerate(x_coefs):
        exp = int(i/n)
        x_poly[i%n,j].coef[exp] = c

    return x_poly

@dataclass
class Eigen():
    ''' 
    Data class for generalized eigenvalues/eigenvectors.
    Holds: - (α, β) polar form of eigenvalue
           - left/right eigenvectors
           - eigenvalue inertia, if real
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
            powers = [ int(p) for p in coef.keys() if not int(p) in powers ]
            self.num_coef = len(powers)
            self.ndim = coef[powers[0]].ndim
            self.shape = coef[powers[0]].shape
        elif isinstance(coef, np.ndarray):
            degrees = [ c.degree() for index, c in np.ndenumerate(coef) ]
            self.num_coef = max(degrees) + 1
            self.ndim = coef.ndim
            self.shape = coef.shape
        else:
            raise TypeError("MatrixPolynomial must receive a list/dict of coefficients.")

        self.vector_like = self.ndim == 1
        self.matrix_like = self.ndim == 2
        self.is_square = self.matrix_like and (self.shape[0] == self.shape[1])

        if self.is_square: 
            self.type = 'regular'
        else: 
            self.type = 'singular'

        # Initializes coefficients and polynomial array
        self.coef = [ np.zeros(self.shape) for _ in range(self.num_coef) ]
        self.poly_array: np.ndarray[Poly] = np.zeros(self.shape, dtype=Poly)
        for index, _ in np.ndenumerate(self.poly_array):
            self.poly_array[index] = Poly([0.0], symbol=self.symbol)

        # Sets coefficients
        self.update(coef=coef)

        ''' Parameters '''
        self.realEigenTol = 1e-10           # Tolerance to consider an eigenvalue as real
        self.max_order = 2                 # Max. polynomial order to compute nullspace solutions

        ''' Setups the SOS decomposition problem in CVXPY '''
        self._init_sos_decomposition()

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
        ret_str = f"({self.shape[0]}"
        ret_str += "".join([ f"x{self.shape[dim]}" for dim in range(1,self.ndim) ])
        ret_str += "".join([ ')-dim {}'.format(type(self).__name__), f" on {self.symbol}: P({self.symbol}) = P₀"])
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

    def _compute_sos_locs(self):
        '''
        Computes the index matrix representing the rule for placing the coefficients in the correct places on the 
        shape matrix of the SOS representation. Algorithm gives preference for putting the elements of coeffs 
        closer to the main diagonal of the SOS matrix.
        '''
        # Compute SOS kernel first
        self.sos_kern_deg = math.ceil(self.degree/2)
        self.sos_locs = [ [] for _ in self.coef ]
        for exp in range(self.num_coef):

            # Checks the possible (i,j) locations on SOS matrix where the monomial can be put
            for (i,j) in product(range(self.sos_kern_deg+1),range(self.sos_kern_deg+1)):
                if i > j: continue
                if exp == i + j:
                    self.sos_locs[exp].append( (i,j) )

    def _init_sos_decomposition(self):
        ''' Initialization of SOS decomposition '''

        if not self.type == "regular":
            return
        
        # if self.degree % 2 != 0:
        #     warnings.warn("Be aware that a polynomial matrix of odd degree can never be p.s.d.")

        self._compute_sos_locs()
        dim = self.shape[0]
        self.sos_dim = (self.sos_kern_deg + 1) * dim

        ''' Setup CVXPY parameters and variables '''
        self.coef_params = [ cvx.Parameter( shape=self.shape ) for _ in self.coef ]
        self.SOS = cvx.Variable( (self.sos_dim, self.sos_dim), symmetric=True )
        self.SOSt = np.zeros((self.sos_kern_deg+1, self.sos_kern_deg+1), dtype=object)
        for i in range(self.sos_kern_deg+1):
            for j in range(self.sos_kern_deg+1):
                if i <= j:
                    self.SOSt[i,j] = cvx.Variable( shape=self.shape, symmetric=True if i == j else False )
                else:
                    self.SOSt[i,j] = self.SOSt[j,i].T

        ''' Setup CVXPY cost, constraints and problem '''
        self.constraints = [ self.SOS >> 0 ]
        for locs, c_param in zip(self.sos_locs, self.coef_params):
            self.constraints += [ sum([ self.SOSt[index] if index[0]==index[1] else self.SOSt[index] + self.SOSt[index].T for index in locs ]) == c_param ]
        self.cost = cvx.norm( self.SOS - cvx.bmat(self.SOSt.tolist()) )
        self.sos_problem = cvx.Problem( cvx.Minimize( self.cost ), self.constraints )

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

    def _symm(self, type = +1) -> np.ndarray[Poly]:
        '''
        Returns equivalent symmetric/antisymmetric matrix polynomial (or linear matrix pencil).
        '''
        if not self.is_square:
            if type == +1: type_text = "symmetric"
            if type == -1: type_text = "antisymmetric"
            msg_txt = f"Cannot compute {type_text} part of non-square matrix polynomial."
            raise Exception(msg_txt)
        
        return 0.5 * ( self.poly_array + type * self.poly_array.T )

    def _test_sos_decomposition(self, num_samples):
        '''
        Test method for SOS decomposition.
        '''
        sos_matrix = self.sos_decomposition(verbose=True)

        def call(l):
            Lambda = np.vstack([ (l**k)*np.eye(*self.shape) for k in range(self.sos_kern_deg+1) ])
            return Lambda.T @ sos_matrix @ Lambda

        error = 0.0
        for _ in range(num_samples):
            l = np.random.randn()
            val1 = self(l)
            val2 = call(l)
            error += np.linalg.norm( val1 - val2 )

        print(f"Test for SOS decomposition finished after {num_samples} random trials with error = {error}.")

    def outer(poly1, poly2):
        ''' Outer product between MatrixPolynomials '''
        return MatrixPolynomial._multiply(poly1, poly2, type = 0)

    def update(self, coef: list | dict | np.ndarray):
        '''
        Update method for matrix polynomial. Existing coefficients can be modified, 
        but the polynomial shape cannot be changed after creation.

        Input: - list [ coef0, coef1, ... , coefN ] of ordered powers OR
               - dict { 0: coef0, 1: coef1, ... , N: coefN } of (power:coef) pairs 
                        (repeating power keys are not allowed and return an error)
               - np.ndarray[numpy.polynomial.Polynomial]
        '''
    
        ''' (Default) - updates by ordered sequence of increasing powers (numpy Polynomial style) '''
        if isinstance(coef, list):
            self.coef = []
            for c in coef:
                self._verify_coef(c)
                self.coef.append(c)
            self.poly_array = to_array(self.coef, self.symbol)

        ''' Updates by dictionary of (power:coefficient) '''
        if isinstance(coef, dict):
            powers = []
            self.coef = []
            for k, c in coef.items():
                self._verify_coef(c)

                # Verify power entries
                if ( isinstance(k, int) and k >= 0 ) or ( isinstance(k, str) and str.isnumeric(k) and int(k) >= 0 ):
                    k = int(k)
                    if k not in powers: 
                        powers.append(k)
                    else:
                        raise TypeError("Duplicate entry of same power.")
                else:
                    raise TypeError("Dictionary keys must represent polynomial powers.")
                
                # appends new coefficients if necessary
                if k >= len(self.coef):
                    num_extra_coefs = k - len(self.coef) + 1
                    self.coef += [ np.zeros(*( self.shape[i] for i in range(self.ndim))) for _ in range(num_extra_coefs) ]

                # update coefficient k
                self.coef[k] = c
            self.poly_array = to_array(coef, self.symbol)

        ''' Updates by np.ndarray of numpy Polynomials '''
        if isinstance(coef, np.ndarray):
            self.coef, symbol = to_coef(coef)
            self.poly_array = to_array(self.coef, symbol)

        # at this point, self.coef and self.poly_array are updated

        ''' Updates the number of coefs, max degree and reinitiaze sos decomposition if necessary '''
        old_num_coef = self.num_coef
        self.num_coef = len(self.coef)
        self.degree = self.num_coef - 1
        if self.num_coef != old_num_coef:
            self._init_sos_decomposition()

    def nullspace(self) -> np.ndarray[Poly]:
        ''' 
        Returns the minimum (right) nullspace polynomial of P(λ), that is,
        a new polynomial matrix N(λ) = N0 + λ N1 + λ² N2 + ... of P(λ) satisfying P(λ) N(λ) = 0.

        Inputs: - max_degree is the maximum possible degree of N(λ)
        '''
        return nullspace(self.poly_array, self.symbol)

    def symmetric(self) -> np.ndarray[Poly]:
        '''
        Returns equivalent symmetric matrix polynomial.
        '''
        return self._symm(type=+1)

    def antisymmetric(self) -> np.ndarray[Poly]:
        '''
        Returns equivalent antisymmetric matrix polynomial.
        '''
        return self._symm(type=-1)

    def companion_form(self):
        ''' 
        Returns a MatrixPencil that is the equivalent companion form of P(λ).
        '''
        M, N = companion_form( self.coef )
        return MatrixPencil( M, N )

    def sos_decomposition(self, verbose=False):
        '''
        SOS decomposition of MatrixPolynomial
        '''
        for k, c in enumerate(self.coef):
            self.coef_params[k].value = c

        self.sos_problem.solve(solver = 'SCS', verbose=False)
        if verbose: 
            print(f"SOS decomposition problem returned with status {self.sos_problem.status}, final cost = {self.cost.value}")

        sos_matrix = np.block([[ self.SOSt[i,j].value for j in range(self.sos_kern_deg+1) ] for i in range(self.sos_kern_deg+1) ])
        return sos_matrix

    @property
    def T(self):
        ''' Transpose operation '''
        return self.poly_array.T

    @classmethod
    def constant(cls, const_arr: np.ndarray):
        '''
        Returns a constant MatrixPolynomial with only one coefficient const_arr
        '''
        if not isinstance(const_arr, np.ndarray):
            raise TypeError("Input must be a constant numpy array.")

        for index, ele in np.ndenumerate(const_arr):
            if not isinstance(ele, (float, int, np.float32, np.int32)):
                raise TypeError("Input must be a constant numpy array.")
        
        if cls == MatrixPolynomial:
            return cls(coef=[ const_arr ])
        
        if cls == MatrixPencil:
            return cls( M=np.zeros(const_arr.shape), N=-const_arr )

    @classmethod
    def eye(cls, dim=2):
        '''
        Returns the identity MatrixPolynomial of dimension dim
        '''
        return cls.constant( np.eye(dim) )

    @classmethod
    def zeros(cls, size=(2,2)):
        '''
        Returns the identity MatrixPolynomial of dimension dim
        '''
        return cls.constant( np.zeros(size) )

    @classmethod
    def diag(cls, poly_list: list):
        '''
        Returns a diagonal MatrixPolynomial with the numpy Polynomials in poly_list
        '''
        if not isinstance(poly_list, list):
            raise TypeError("Input must be a list of numpy Polynomials.")

        return cls(coef=np.diag(poly_list))

class MatrixPencil(MatrixPolynomial):
    '''
    Class for linear matrix pencils of the form P(λ) = λ M - N (derived from MatrixPolynomial class)
    Each object holds: - its M, N matrices;
                       - a list with all pencil eigenvalues in standard/polar form and corresponding left/right eigenvectors;
    '''
    def __init__(self, M: list | np.ndarray, N: list | np.ndarray, **kwargs):

        if isinstance(M, list): M = np.array(M)
        if isinstance(N, list): N = np.array(N)
        
        if M.ndim != 2 or N.ndim != 2:
            raise TypeError("M and N must be two dimensional arrays.")

        if M.shape != N.shape:
            raise TypeError("Matrix dimensions are not equal.")
        
        self.N, self.M = N, M
        super().__init__(coef=[ -N, M ], **kwargs)

        '''
        Uses QZ algorithm to decompose the two pencil matrices into M = Q MM Z' and N = Q NN Z',
        where MM is block upper triangular, NN is upper triangular and Q, Z are unitary matrices.
        '''
        if self.type == 'regular':
            self.NN, self.MM, self.alphas, self.betas, self.Q, self.Z = sp.linalg.ordqz(self.N, self.M, output='real')
            self.eigens = self._eigen(self.alphas, self.betas)

    def __call__(self, alpha: int | float, beta: int | float = 1.0) -> np.ndarray:
        '''
        Returns pencil value.
        If only one argument is passed, it is interpreted as the λ value and method returns matrix P(λ) = λ M - N.
        If two arguments are passed, they are interpreted as α, β values and method returns P(α, β) = α M - β N.
        '''
        if beta != 1:
            return alpha * self.M - beta * self.N
        else:
            return super().__call__(alpha)

    def _blocks(self) -> tuple[ list[Poly], list[np.ndarray[Poly]] ]:
        '''  
        Computes information about blocks of the QZ decomposition.
        Returns: - list of block poles
                 - list of block adjoint matrices
        '''
        n = self.M.shape[0]

        ''' Computes the block poles and adjoint matrices '''
        blk_poles, blk_adjs = [], []
        i = 0
        while i < n:
            # 2X2 BLOCKS OF COMPLEX CONJUGATE PENCIL EIGENVALUES
            if i < n-1 and self.NN[i+1,i] != 0.0:

                MMblock = self.MM[i:i+2,i:i+2]      # this is diagonal
                NNblock = self.NN[i:i+2,i:i+2]      # this is full 2x2

                a = np.linalg.det(MMblock)
                b = -( MMblock[0,0] * NNblock[1,1] + MMblock[1,1] * NNblock[0,0] )
                c = np.linalg.det(NNblock)
                blk_poles.append( Poly([ c, b, a ], symbol=self.symbol) )

                adj11 = Poly([ -NNblock[1,1],  MMblock[1,1] ], symbol=self.symbol)
                adj12 = Poly([ NNblock[0,1] ], symbol=self.symbol)
                adj21 = Poly([ NNblock[1,0] ], symbol=self.symbol)
                adj22 = Poly([ -NNblock[0,0],  MMblock[0,0] ], symbol=self.symbol)
                blk_adjs.append( np.array([[ adj11, adj12 ],[ adj21, adj22 ]]) )

                i+=2
            # 1X1 BLOCKS OF REAL PENCIL EIGENVALUES
            else:
                MMblock = self.MM[i,i]
                NNblock = self.NN[i,i]

                blk_poles.append( Poly([ -NNblock, MMblock ], symbol=self.symbol) )
                blk_adjs.append( np.array([Poly(1.0, symbol=self.symbol)]) )

                i+=1
                
        return blk_poles, blk_adjs

    def _eigen(self, alphas: np.ndarray, betas: np.ndarray) -> list[Eigen]:
        '''
        Computes generalized eigenvalues/eigenvectors from polar eigenvalues.
        '''
        if len(alphas) != len(betas):
            raise TypeError("The same number of polar eigenvalues must be passed.")

        eigens: list[Eigen] = []
        for alpha, beta in zip(alphas, betas):

            P = self(alpha, beta)
            zRight = null_space(P, rcond=1e-11)
            zLeft = null_space(P.T, rcond=1e-11)

            zRight = zRight.reshape(self.shape[0], )
            zLeft = zLeft.reshape(self.shape[0], )

            inertia = zLeft.T @ self.M @ zRight

            if beta != 0:
                eigens.append( Eigen(alpha, beta, alpha/beta, zRight, zLeft, inertia) )
            else:
                eigens.append( Eigen(alpha, 0.0, np.inf if alpha.real > 0 else -np.inf, zRight, zLeft, inertia) )

        return eigens

    def real_eigen(self) -> list[Eigen]:
        '''
        Returns an Eigen list with sorted real eigenvalues
        '''
        realAlphas, realBetas = [], []
        for eig in self.eigens:
            if np.abs(eig.alpha.imag) < self.realEigenTol:
                realAlphas.append( eig.alpha.real )
                realBetas.append( eig.beta )

        realEigens = self._eigen(realAlphas, realBetas)
        realEigens.sort(key=lambda eig: eig.eigenvalue)

        return realEigens

    def inverse(self) -> tuple[ Poly, np.ndarray[Poly] ]:
        '''
        Returns: - polynomial determinant det(λ)
                 - pencil adjoint polynomial matrix adj(λ)
        Used to compute the pencil inverse P(λ)^(-1) = det(λ)^(-1) adj(λ).

        TO DO (in the future): generalize the computation of the inverse for general MatrixPolynomials
        '''
        if not self.is_square:
            raise Exception("Inverse is not defined for non-square matrix polynomials.")

        n, m = self.shape[0], self.shape[1]

        ''' Computes blocks of the QZ decomposition '''
        blk_poles, blk_adjs = self._blocks()
        blk_dims = [ pole.degree() for pole in blk_poles ]

        ''' Computes pencil determinant '''
        determinant = np.prod(blk_poles)

        ''' Computes the pencil adjoint matrix '''
        num_blks = len(blk_poles)
        adjoint_arr = np.array([[ Poly([0.0], symbol=self.symbol) for _ in range(n) ] for _ in range(n) ])

        # Iterate over each block, starting by the last one
        for i in range(num_blks-1, -1, -1):
            blk_i_slice = slice( sum(blk_dims[0:i]), sum(blk_dims[0:i+1]) )

            for j in range(i, num_blks):
                blk_j_slice = slice( sum(blk_dims[0:j]), sum(blk_dims[0:j+1]) )

                ''' 
                j == i: Computes ADJOINT DIAGONAL BLOCKS
                j != i: Computes ADJOINT UPPER TRIANGULAR BLOCKS
                '''
                if j == i:
                    poles_ij = np.array([[ np.prod([ pole for k, pole in enumerate(blk_poles) if k != j ]) ]])
                    Lij = poles_ij * blk_adjs[j]
                else:
                    Tii = self.MM[ blk_i_slice, blk_i_slice ]
                    Sii = self.NN[ blk_i_slice, blk_i_slice ]

                    b_poly = np.array([[ Poly([0.0], symbol=self.symbol) for _ in range(blk_dims[j]) ] for _ in range(blk_dims[i]) ])
                    for k in range(i+1, j+1):
                        blk_k_slice = slice( sum(blk_dims[0:k]), sum(blk_dims[0:k+1]) )

                        # Compute polynomial (λ Tik - Sik) and get the kj slice of adjoint
                        Tik = self.MM[ blk_i_slice, blk_k_slice ]
                        Sik = self.NN[ blk_i_slice, blk_k_slice ]
                        poly_ik = np.array([[ Poly([ -Sik[a,b], Tik[a,b] ], symbol=self.symbol) for b in range(Tik.shape[1]) ] for a in range(Tik.shape[0]) ])
                        adjoint_kj = adjoint_arr[ blk_k_slice, blk_j_slice ]

                        b_poly -= poly_ik @ adjoint_kj

                    Lij = solve_poly_linearsys( Tii, Sii, b_poly )

                # Populate adjoint matrix
                adjoint_arr[ blk_i_slice, blk_j_slice ] = Lij

        adjoint = self.Z @ adjoint_arr @ self.Q.T

        return determinant, adjoint

    def update(self, **kwargs):
        ''' 
        Update method for Linear Matrix Pencil. Existing coefficients can be modified, 
        but the pencil shape cannot be changed after creation.

        Inputs: - M np.array
                - N np.array        for pencil P(λ) = λ M - N
        '''
        N, M = self.N, self.M
        for key in kwargs.keys():
            if key == 'M':
                M = kwargs['M']
                continue
            if key == 'N':
                N = kwargs['N']
                continue

        super().update(coef=[-N, M])
        self.N, self.M = -self.coef[0], self.coef[1]

        # Recomputes eigenvalues
        if self.type == 'regular':
            self.NN, self.MM, self.alphas, self.betas, self.Q, self.Z = sp.linalg.ordqz(self.N, self.M, output='real')
            self.eigens = self._eigen(self.alphas, self.betas)