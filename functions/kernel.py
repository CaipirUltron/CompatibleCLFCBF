from math import comb
from scipy.optimize import minimize

from common import *
from dynamic_systems import Integrator
from .basic import Function, MultiPoly, LeadingShape, commutation_matrix, mat

import logging
import warnings
import itertools
import numpy as np
import scipy as sp
import cvxpy as cp
import sympy as sym

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
        self._compute_D()

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
        self.Aops = [ Op(A) for A in self.A ]

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

    def _compute_D(self):
        '''
        Given a vector function Φ(x) = N m(x), Φ: Rn -> Rn, this method computes the SOS factorized form 
        of the determinant of the Jacobian, that is, |∇Φ(x)| = mΦ(x)' D(N) mΦ(x) for some kernel mΦ(x).
        Computes: self.D is a symmetric matrix lambda function of matrix N (Rn x Rp).
        Application: generation of invex CLF/CBFs.
        '''
        N = sym.MatrixSymbol("N", self._dim, self._num_monomials)
        N_list = [ N @ Ai for Ai in self.get_A_matrices() ]    # list of n x p

        M_list = []
        for k in range(self._num_monomials):
            Mk = np.zeros([self._dim, self._dim], dtype=N_list[0].dtype )
            for i, Ni in enumerate(N_list):
                Mk[:,i] = Ni[:,k]
            M_list.append(Mk)

        # ------------------------ Generation and SOS factorization of |JΦ(x)| -----------------------------

        ''' Compute the Jacobian JΦ(x) of the vector function Φ(x) = N m(x) and its determinant |JΦ(x)| '''
        delPhi = MultiPoly(self._powers, M_list).filter()
        det = delPhi.determinant()
        self.det_coeffs_fun = sym.lambdify( N, det.coeffs )

        ''' Compute suitable kernel mΦ(x) for SOS factorization of |JΦ(x)| '''
        self.det_kernel = det.sos_kernel()
        self.dim_det_kernel = len(self.det_kernel)

        '''
        Compute the symbolic matrix function D(N) for SOS form of |JΦ(x)|, that is, 
        |JΦ(x)| = mΦ(x)' D(N) mΦ(x), and its corresponding lambda function for numeric computation.
        '''
        sos_index_matrix = det.sos_index_matrix(self.det_kernel)
        self.Dfun_symbolic = det.shape_matrix(self.det_kernel, sos_index_matrix)
        self._Dfun = sym.lambdify( N, self.Dfun_symbolic ) # >> 0

        ''' Compute the matrix derivatives of D(N) with respect to each Nij variable, and corresponding lambda functions '''
        self.Dfun_symbolic_derivatives = [ [ None for _ in range(self._num_monomials) ] for _ in range(self._dim) ]
        self._Dfun_derivatives = [ [ None for _ in range(self._num_monomials) ] for _ in range(self._dim) ]
        for i, j in itertools.product( range(self._dim), range(self._num_monomials) ):

            Derivative = [ [ None for _ in range(self.dim_det_kernel) ] for _ in range(self.dim_det_kernel) ]
            for k, l in itertools.product( range(self.dim_det_kernel), range(self.dim_det_kernel) ):
                Derivative[k][l] = sym.diff( self.Dfun_symbolic[k][l], N[i,j] )

            self.Dfun_symbolic_derivatives[i][j] = Derivative
            self._Dfun_derivatives[i][j] = sym.lambdify( N, Derivative )

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

    def get_diff_operators(self):
        '''
        Return the diff. operators
        '''
        return self.Aops

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

    def D(self, N):
        if N.shape != (self._dim, self._num_monomials):
            raise Exception("Passed N matrix has incorrect dimensions.")
        return np.array(self._Dfun(N))

    def D_partials(self, N):
        if N.shape != (self._dim, self._num_monomials):
            raise Exception("Passed N matrix has incorrect dimensions.")
        
        q = self.dim_det_kernel
        Dfun_derivatives = [ [ np.zeros((q,q)) for _ in range(self._num_monomials) ] for _ in range(self._dim) ]
        for i, j in itertools.product( range(self._dim), range(self._num_monomials) ):
            Dfun_derivatives[i][j] = np.array( self._Dfun_derivatives[i][j](N) )

        return Dfun_derivatives

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
        is_scalar = are_all_type(coeffs, (int, float))
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

    def get_kernel(self):
        ''' Returns the monomial basis vector '''
        return self.kernel

    def to_poly(self):
        ''' Returns corresponding MultiPoly object '''
        return MultiPoly(self.kernel._powers, self.coeffs)

    @classmethod
    def from_poly(cls, poly: MultiPoly):
        ''' Constructor for creating KernelLinear from a MultiPoly '''

        if isinstance(poly.coeffs[0], sym.Expr):
            raise Exception("Simbolic coefficients are not supported.")

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

        self.cost_functions = []
        self.eq_constraint_functions = []
        self.ineq_constraint_functions = []
        self.fit_options = { "invex_fit": False, 
                             "invex_constraint": True }

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

        # init_shape = kernel_quadratic(eigen=np.ones(self._dim), R=np.eye(self._dim), center=np.zeros(self._dim), kernel_dim=self.kernel_dim)
        init_shape = np.random.randn(self.kernel_dim, self.kernel_dim)
        init_shape = np.zeros((self.kernel_dim, self.kernel_dim))
        self.fit_options["initial_shape"] = init_shape.T @ init_shape
        
        tol = 0e-2
        q = self.kernel.dim_det_kernel
        self.Tol = np.zeros((q,q))
        self.Tol[0,0] = 1
        self.Tol = tol*self.Tol

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
        ''' Setting method for the shape matrix. '''

        # If a one dimensional array was passed, checks if it can be converted to symmetric matrix
        if shape_matrix.ndim == 1:
            roots = np.roots([1, 1, -2*len(shape_matrix)])
            if np.any([ root.is_integer() and root > 0 for root in roots ]):
                shape_matrix = vector2sym(shape_matrix.tolist())
            else:
                raise Exception("Number of coefficients is not compatible with a symmetric matrix.")

        if shape_matrix.shape[0] != shape_matrix.shape[1]:
            raise Exception("Matrix of coefficients must be a square.")
        
        if np.linalg.norm( shape_matrix - shape_matrix.T ) > 1e-3:
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

        if "initial_shape" in keys:
            self.fit_options["initial_shape"] = kwargs["initial_shape"]

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
            if self.fit_options["invex_fit"]:
                fitted_shape = self.invex_fitting()
            else:
                fitted_shape = self.convex_fitting()
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
                self.cost_functions += [ lambda N, G : ( self._fun(point["coords"], N.T @ G @ N) - point["level"] )**2 ]

        # Define gradient constraints
        if "gradient" in keys:
            gradient_norm = cp.Variable()
            gradient = np.array(point["gradient"])
            normalized = gradient/np.linalg.norm(gradient)

            self.constraints += [ gradient_norm >= 0 ]
            if self.force_gradients or point["force_gradient"]:
                self.constraints += [ self._grad(point["coords"], self.SHAPE) == gradient_norm * normalized ]
            else:
                if isinstance(self, KernelLyapunov):
                    self.cost += cp.norm( self._grad(point["coords"], self.SHAPE) - gradient_norm * normalized )
                else:
                    self.cost += cp.norm( self._grad(point["coords"], self.SHAPE) - gradient_norm * normalized )
            
            gradient = np.array(point["gradient"])
            normalized = gradient/np.linalg.norm(gradient)
            self.cost_functions += [ lambda N, G : ( normalized.T @ self._grad(point["coords"], N.T @ G @ N) - np.linalg.norm(self._grad(point["coords"], N.T @ G @ N)) )**2 ]

        # Define curvature constraints (2D only)
        if "curvature" in keys:
            if self._dim != 2:
                raise Exception("Error: curvature fitting was not implemented for dimensions > 2. ")
            if "gradient" not in keys:
                raise Exception("Cannot specify a curvature without specifying the gradient.")

            v = rot2D(np.pi/2) @ normalized
            self.cost += ( self._hessian_quadratic_form(point["coords"], self.SHAPE, v) - point["curvature"] )**2
            self.cost_functions += [ lambda N, G : ( v.T @ self._hess(point["coords"], N.T @ G @ N) @ v - point["curvature"] )**2 ]

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
            if contained: 
                self.constraints.append( self._fun(pt, self.SHAPE) <= level )
                self.ineq_constraint_functions += [ lambda N, G: level - self._fun(pt, N.T @ G @ N ) ]

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

            inner_cvxpy = (+1 if increasing else -1) * ( next_pt - curr_pt ).T @ self._grad(curr_pt, self.SHAPE)
            self.constraints.append( self._fun(next_pt, self.SHAPE) - self._fun(curr_pt, self.SHAPE) >= inner_cvxpy )

            inner = lambda N, G: (+1 if increasing else -1) * ( next_pt - curr_pt ).T @ self._grad(curr_pt, N.T @ G @ N)
            self.ineq_constraint_functions += [ lambda N, G : self._fun(next_pt, N.T @ G @ N) - self._fun(curr_pt, N.T @ G @ N) - inner(N,G) ]
                
    def add_skeleton_constraints(self, skeleton_segments):
        '''
        Generates the appropriate constraints for smooth increasing of the CBF from a center point located on the skeleton curve.
        Parameters: - skeleton_segments is an array with segments, each containing sampled points of the obstacle medial-axis.
                    - the points on each segment are assumed to be ordered: that is, the barrier must grow from one point to the next
        '''
        for segment in skeleton_segments:
            self.add_center_constraints(point_list=segment)
            self.add_continuity_constraints(segment, increasing=True)

    def convex_fitting(self):
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

    def reshape(self, var: np.ndarray):
        ''' Converts from array var to tuple of matrices N and G '''
        n = self._dim
        p = self.kernel_dim

        N_arr = var[0:n*p]
        sqG_arr = var[n*p:]

        N = N_arr.reshape((n,p))
        sqG = sqG_arr.reshape((n,n))
        G = sqG.T @ sqG

        return N, G

    def flatten(self, N, G):
        ''' Converts from N, G to var array '''

        N_arr = N.flatten()
        sqG = sp.linalg.sqrtm(G)
        sqG_arr = sqG.flatten()
        var = np.hstack([N_arr, sqG_arr])

        return var

    def invex_fitting(self):
        ''' 
        Nonconvex optimization problem for fitting an invex function to point-like constraints.
        Must be correctly initialized by an invex function.
        Returns: an invex P shape
        '''
        initial_shape = self.fit_options["initial_shape"]
        Ninit, Ginit, _ = NGN_decomposition(self._dim, initial_shape)
        init_eigs = np.linalg.eigvals(Ginit)
        eig_upperbound, eig_lowerbound = np.max(init_eigs), np.min(init_eigs)

        print(f"Initial eigenvalues = {init_eigs}")

        if not np.all( np.linalg.eigvals(Ginit) >= 0):
            raise Exception("Initial P must be psd.")

        Dinit = self.kernel.D(Ninit)
        dist_to_psd = np.linalg.norm( PSD_closest(Dinit) - Dinit )
        dist_to_nsd = np.linalg.norm( NSD_closest(Dinit) - Dinit )

        if dist_to_psd <= dist_to_nsd: 
            cone = +1
            print(f"D(Ninit) closer to PSD cone.")
        else: 
            cone = -1
            print(f"D(Ninit) closer to NSD cone.")

        def objective(var: np.ndarray) -> float:

            N, G = self.reshape(var)
            
            cost = 0.0
            # cost += invexity_constr(var)
            cost += sum([ cost_fun(N,G) for cost_fun in self.cost_functions ])
            return cost

        def eq_constr(var: np.ndarray) -> list[float]:

            N, G = self.reshape(var)
            return [ fun(N,G) for fun in self.eq_constraint_functions ]

        def ineq_constr(var: np.ndarray) -> list[float]:

            N, G = self.reshape(var)
            return [ fun(N,G) for fun in self.ineq_constraint_functions ]

        def invexity_constr(var: np.ndarray) -> float:

            N, G = self.reshape(var)            
            D = self.kernel.D(N)

            if cone == +1:
                ProjD = PSD_closest(D-self.Tol)
            if cone == -1:
                ProjD = NSD_closest(D+self.Tol)

            return np.linalg.norm(D - ProjD)

        def orthonormality_constr(var: np.ndarray) -> float:
            ''' Keeps N orthonormal '''

            N, G = self.reshape(var)
            return np.linalg.norm( N @ N.T - np.eye(self._dim) )

        def eig_bounds_constr(var: np.ndarray) -> list[float]:
            ''' Keeps eigenvalues of P well-behaved '''

            N, G = self.reshape(var)
            eigs = np.diag(G)

            # eig_upperbound, eig_lowerbound = np.max(init_eigs), np.min(init_eigs)
            eig_upperbound, eig_lowerbound = 10.0, 0.0
            eig_bounds = np.array([ eig_upperbound - max(eigs), min(eigs) - eig_lowerbound ])

            return np.hstack([eig_bounds])

        def intermediate(res):
            
            N, G = self.reshape(res)
            print(F"N = {N}")
            print(F"G = {G}")

        constraints = []
        constraints += [ {"type": "ineq", "fun": eig_bounds_constr} ]
        constraints += [ {"type": "eq", "fun": eq_constr} ]
        constraints += [ {"type": "ineq", "fun": ineq_constr} ]

        constraints += [ {"type": "eq", "fun": orthonormality_constr} ]
        # if self.fit_options["invex_constraint"]:
        #     constraints += [ {"type": "eq", "fun": invexity_constr} ]

        init_var = self.flatten(Ninit, Ginit)
        sol = minimize( objective, init_var, constraints=constraints, options={"disp": True, "maxiter": 1000} )
        print(sol.message)

        Nsol, Gsol =  self.reshape(sol.x)
        Psol = Nsol.T @ Gsol @ Nsol
        
        total_cost = objective(sol.x)
        print(f"Total cost = {total_cost}")
        print(f"Invex cost = {invexity_constr(sol.x)}")

        invexity = np.linalg.eigvals( self.kernel.D(Nsol) )
        print(f"Fitting invexity = {invexity}")

        eigs = np.linalg.eigvals( Gsol )
        print(f"Fitting eigenvalues = {eigs}")

        print(f"Orthonormality of N = {orthonormality_constr(sol.x)}")

        return Psol

    def update(self, param_ctrl, dt):
        '''
        Integrates and updates parameters
        '''
        self.dynamics.set_control(param_ctrl)
        self.dynamics.actuate(dt)
        self.set_params( coefficients = self.dynamics.get_state() )   
