import math
import warnings

import numpy as np
from numpy.polynomial import Polynomial as Poly

from matplotlib.axes import Axes
import matplotlib.patches as patches
import matplotlib.colors as mcolors

import scipy as sp
from scipy import signal
from scipy.optimize import fsolve, minimize
from scipy.linalg import null_space, inv
from dataclasses import dataclass
from itertools import product

from dynamic_systems import DynamicSystem, LinearSystem
from functions import MultiPoly

def to_coefs(poly: np.ndarray):
    ''' Get the coefficients of a given ndarray of Polynomials '''

    if not isinstance(poly, np.ndarray):
        raise TypeError("Polynomial must be an ndarray.")

    v_coefs = []
    for index, vi in np.ndenumerate(poly):

        if not isinstance(vi, Poly):
            raise TypeError("The elements of v(λ) must be polynomials.")
        
        # Creates new coefficients as necessary
        if len(vi.coef) > len(v_coefs):
            for _ in range(len(vi.coef) - len(v_coefs)):
                v_coefs.append( np.zeros(poly.shape) )

        # Populate given coefficient
        for k, c in enumerate(vi.coef):
            v_coefs[k][index] = c

    return v_coefs

def to_poly_array(coefs: list[np.ndarray], symbol='x'):
    ''' From a list of polynomial coefficients, return corresponding ndarray of Polynomials '''

    shape = coefs[0].shape
    poly_arr = np.zeros(shape, dtype=Poly)
    for index, _ in np.ndenumerate(poly_arr):
        poly_arr[index] = Poly([ c[index] for c in coefs ], symbol=symbol)

    return poly_arr

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

def poly_nullspace(Varr: np.ndarray[Poly], max_degree=20):
    ''' 
    Returns the minimum (right) nullspace polynomial N(λ) = N0 + λ N1 + λ² N2 + ... of V(λ), that is, V(λ) N(λ) = 0.
    Inputs:
        - V(λ) is an (n x m) array polynomial given by Varr.
        - max_degree is the maximum degree to try to find N(λ)
    '''
    if Varr.shape == (Varr.shape[0],):
        Varr = Varr.reshape(-1,1)

    n, m = Varr.shape
    if n == m:
        raise Exception("Vmatrix does not have a non-trivial right nullspace.")

    if not isinstance(Varr, np.ndarray):
        raise TypeError("v(λ) must be a ndarray of Polynomials.")

    Vcoefs = to_coefs(Varr)
    Vdegree = len(Vcoefs)-1
    
    for Qdegree in range(0, max_degree+1):

        Vmatrix = np.zeros([ n*(Vdegree + Qdegree + 1), m*(Qdegree+1) ])
        
        sliding_list = [ c for c in Vcoefs ]
        zeros = [ np.zeros((n,m)) for _ in range(Qdegree) ]
        sliding_list = zeros + sliding_list + zeros
        
        for i in range(Vdegree + Qdegree + 1):
            l = sliding_list[i:i+Qdegree+1]
            l.reverse()
            Vmatrix[i*n:(i+1)*n,:] = np.hstack(l)

        Null = sp.linalg.null_space(Vmatrix)
        if Null.size != 0:
            break

        if Qdegree == max_degree:
            warnings.warn("Vmatrix likely does not have a non-trivial right nullspace.")
            return None

    # Qarr = np.hstack([ ( Null @ params[k*q:(k+1)*q] ).reshape(-1,1) for k in range(p) ])

    Ncoefs = [ Null[i*m:(i+1)*m,:] for i in range(Qdegree+1) ]
    return to_poly_array(Ncoefs, symbol=Varr[0,0].symbol)

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

class MatrixPencil():
    '''
    Class for linear matrix pencils of the form P(λ) = λ M - N.
    - generalized eigenvalues/eigenvector pairs
    - generate equivalent symmetric pencil
    '''
    def __init__(self, M: list | np.ndarray, N: list | np.ndarray):

        if isinstance(M, list): M = np.array(M)
        if isinstance(N, list): N = np.array(N)
        
        if M.ndim != 2 or N.ndim != 2:
            raise TypeError("M and N must be two dimensional arrays.")

        if M.shape != N.shape:
            raise TypeError("Matrix dimensions are not equal.")

        self.type = ''
        if M.shape[0] == M.shape[1]: 
            self.type += 'regular'
        else: 
            self.type += 'singular'

        self.M, self.N = M, N
        self.shape = M.shape

        '''
        Uses QZ algorithm to decompose the two pencil matrices into M = Q MM Z' and N = Q NN Z',
        where MM is block upper triangular, NN is upper triangular and Q, Z are unitary matrices.
        '''
        if self.type == 'singular':
            warnings.warn("Eigenvalue computation (currently) not implemented for singular matrix pencils.")
        else:
            self.NN, self.MM, self.alphas, self.betas, self.Q, self.Z = sp.linalg.ordqz(self.N, self.M, output='real')
            self.eigens = self._eigen(self.alphas, self.betas)

        ''' Parameters '''
        self.realEigenTol = 1e-10           # Tolerance to consider an eigenvalue as real
        self.max_order = 10                 # Max. polynomial order to compute nullspace solutions 

    def set(self, **kwargs):
        ''' Pencil update method '''

        newM, newN = self.M, self.N
        for key in kwargs.keys():
            if key == 'M':
                newM = kwargs['M']
                continue
            if key == 'N':
                newN = kwargs['N']
                continue
        self.__init__(newM, newN)

    def get_eigen(self) -> list[Eigen]:
        ''' Get method for generalized eigenvalues '''
        return self.eigens

    def get_real_eigen(self) -> list[Eigen]:
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

    def has_real_spectra(self):
        ''' Checks if the pencil has real spectra '''
        for eig in self.eigens:
            if np.abs(eig.alpha.imag) > self.realEigenTol:
                return False
        return True

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

            # print( np.linalg.eigvals(P) )
            # print(f"zRight = {zRight}")
            # print(f"zLeft = {zLeft}")

            zRight = zRight.reshape(self.shape[0], )
            zLeft = zLeft.reshape(self.shape[0], )

            inertia = zLeft.T @ self.M @ zRight

            if beta != 0:
                eigens.append( Eigen(alpha, beta, alpha/beta, zRight, zLeft, inertia) )
            else:
                eigens.append( Eigen(alpha, 0.0, np.inf if alpha.real > 0 else -np.inf, zRight, zLeft, inertia) )

        return eigens

    def nullspace(self) -> np.ndarray:
        ''' 
        Returns the pencil minimum nullspace polynomial N(λ), 
        a matrix polynomial satisfying ( λ M - N ) N(λ) = 0, identically (for all λ).
        '''
        return poly_nullspace( self.to_poly_array() )

    def symmetric(self):
        ''' Returns equivalent symmetric pencil. '''
        return MatrixPencil( 0.5*(self.M + self.M.T), 0.5*(self.N + self.N.T) )

    def antisymmetric(self):
        ''' Returns equivalent symmetric pencil. '''
        return MatrixPencil( 0.5*(self.M - self.M.T), 0.5*(self.N - self.N.T) )

    def to_poly_array(self):
        ''' Returns equivalent array of polynomials '''
        poly_arr = np.array([[ Poly([0.0], symbol='λ') for j in range(self.shape[1]) ] for i in range(self.shape[0]) ])
        for (i,j) in product(range(self.shape[0]),range(self.shape[1])):
            poly_arr[i,j] += Poly([-self.N[i,j], self.M[i,j] ], symbol='λ')
        return poly_arr

    def inverse(self):
        '''
        Returns:  - pencil adjoint polynomial matrix adj(λ)
                  - polynomial determinant det(λ)
        Used to compute the pencil inverse P(λ)^(-1) = det(λ)^(-1) adj(λ).
        '''
        n, m = self.shape[0], self.shape[1]
        if n != m:
            raise NotImplementedError("Cannot compute the inverse of a non-square matrix pencil.")

        ''' Computes blocks of the QZ decomposition '''
        blk_poles, blk_adjs = self._blocks()
        blk_dims = [ pole.degree() for pole in blk_poles ]

        ''' Computes pencil determinant '''
        determinant = np.prod(blk_poles)

        ''' Computes the pencil adjoint matrix '''
        num_blks = len(blk_poles)
        adjoint_arr = np.array([[ Poly([0.0], symbol='λ') for _ in range(n) ] for _ in range(n) ])

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

                    b_poly = np.array([[ Poly([0.0], symbol='λ') for _ in range(blk_dims[j]) ] for _ in range(blk_dims[i]) ])
                    for k in range(i+1, j+1):
                        blk_k_slice = slice( sum(blk_dims[0:k]), sum(blk_dims[0:k+1]) )

                        # Compute polynomial (λ Tik - Sik) and get the kj slice of adjoint
                        Tik = self.MM[ blk_i_slice, blk_k_slice ]
                        Sik = self.NN[ blk_i_slice, blk_k_slice ]
                        poly_ik = np.array([[ Poly([ -Sik[a,b], Tik[a,b] ], symbol='λ') for b in range(Tik.shape[1]) ] for a in range(Tik.shape[0]) ])
                        adjoint_kj = adjoint_arr[ blk_k_slice, blk_j_slice ]

                        b_poly -= poly_ik @ adjoint_kj

                    Lij = solve_poly_linearsys( Tii, Sii, b_poly )

                # Populate adjoint matrix
                adjoint_arr[ blk_i_slice, blk_j_slice ] = Lij

        return determinant, self.Z @ adjoint_arr @ self.Q.T

    def _blocks(self):
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
                blk_poles.append( Poly([ c, b, a ], symbol='λ') )

                adj11 = Poly([ -NNblock[1,1],  MMblock[1,1] ], symbol='λ')
                adj12 = Poly([ NNblock[0,1] ], symbol='λ')
                adj21 = Poly([ NNblock[1,0] ], symbol='λ')
                adj22 = Poly([ -NNblock[0,0],  MMblock[0,0] ], symbol='λ')
                blk_adjs.append( np.array([[ adj11, adj12 ],[ adj21, adj22 ]]) )

                i+=2
            # 1X1 BLOCKS OF REAL PENCIL EIGENVALUES
            else:
                MMblock = self.MM[i,i]
                NNblock = self.NN[i,i]

                blk_poles.append( Poly([ -NNblock, MMblock ], symbol='λ') )
                blk_adjs.append( np.array([Poly(1.0, symbol='λ')]) )

                i+=1
                
        return blk_poles, blk_adjs

    def __call__(self, alpha: int | float, beta: int | float = 1.0) -> np.ndarray:
        '''
        Returns pencil value.
        If only one argument is passed, it is interpreted as the λ value and method returns matrix P(λ) = λ M - N.
        If two arguments are passed, they are interpreted as α, β values and method returns P(α, β) = α M - β N.
        '''
        return alpha * self.M  - beta * self.N

    def __str__(self) -> str:
        '''
        Print the pencil P(λ) = λ M - N
        '''
        np.set_printoptions(precision=3, suppress=True)
        ret_str = '{}'.format(type(self).__name__) + " = \u03BB M - N with \n"
        ret_str = ret_str + 'M = \n' + self.M.__str__() + '\n'
        ret_str = ret_str + 'N = \n' + self.N.__str__()
        return ret_str

class QFunction():
    ''' 
    General class for Qfunctions q(λ) = v(λ)' H v(λ), where P(λ) v(λ) = w
    and P(λ) is a regular linear matrix pencil.
    '''
    def __init__(self, P: MatrixPencil, H: list | np.ndarray, w: list | np.ndarray):
        
        if isinstance(H, list): H = np.array(H)
        if isinstance(w, list): w = np.array(w)
        self._verify(P, H, w)

        self.pencil = P
        self.symmetricPencil : MatrixPencil = self.pencil.symmetric()

        self.H = H
        self.w = w
        self._compute_polys()

        self.max_order = 100
        self.trim_tol = 1e-8
        self.real_zero_tol = 1e-6
        self.stability_zero_tol = 1e-10

    def _verify(self, P: MatrixPencil, H: list | np.ndarray, w: list | np.ndarray):
        ''' Verification method for passed '''

        if not isinstance( P, MatrixPencil ):
            raise TypeError("P must be a MatrixPencil.")

        if H.ndim != 2:
            raise TypeError("H must be a n x n array.")

        if H.shape[0] != H.shape[1]:
            raise TypeError("H must be a square matrix.")

        self.dim = H.shape[0]

        if len(w) != self.dim:
            raise TypeError("H and w must have the same dimension.")

        if not w.shape in ( (self.dim,), (self.dim,1) ):
            raise TypeError("w must be a n x 1 array.")

    def _compute_polys(self):
        ''' Private method for computing the QFunction polynomials '''

        determinant, adjoint = self.pencil.inverse()
        self.n_poly = ( self.w.T @ adjoint.T @ self.H @ adjoint @ self.w )
        self.d_poly = ( determinant**2 )
        self.zero_poly = ( self.n_poly - self.d_poly )
        self.v_poly, self.det_poly = self._v_poly()
        self.stability_poly = self._stability_poly()

        print(f"S(λ) shape = { self.stability_poly.shape }")
        print(f"S(λ) = { self.stability_poly }")

        if self.stability_poly.shape == (1,1):
            print(f"S(λ) roots = { self.stability_poly[0,0].roots() }")

    def _v_poly_derivative(self, order=0):
        '''
        Computes the derivatives of v(λ) of any order.
        Returns: - vector polynomial v_poly(λ)
                 - scalar divisor polynomial a(λ)
        v(λ) = 1/a(λ) v_poly(λ)
        '''
        det, adjoint = self.pencil.inverse()
        v = math.factorial(order) * adjoint @ self.w

        power = np.eye(self.pencil.shape[0])
        for k in range(order): 
            power @= - adjoint @ self.pencil.M
        v = power @ v

        return v, det**(order+1)

    def _v_poly(self):
        return self._v_poly_derivative(order=0)

    def _stability_poly(self) -> np.ndarray[Poly]:
        ''' Computes the stability polynomial matrix '''

        nablah_poly = self.H @ self.v_poly
        null_poly = poly_nullspace(nablah_poly.reshape(1,-1))

        Ps_poly = self.pencil.symmetric().to_poly_array()
        return null_poly.T @ Ps_poly @ null_poly
        
    def _qvalue(self, l: float, H: list | np.ndarray, w: list | np.ndarray) -> float:
        '''
        Computes the q-function value q(λ) = n(λ)/d(λ) for a given λ (for DEBUG only)
        '''
        P = self.pencil(l)
        v = inv(P) @ self.w
        return v.T @ self.H @ v 

    def set(self, **kwargs):
        ''' QFunction update method '''

        newM, newN = self.pencil.M, self.pencil.N
        newH, new_w = self.H, self.w
        for key in kwargs.keys():
            if key == 'M':
                newM = kwargs['M']
                continue
            if key == 'N':
                newN = kwargs['N']
                continue
            if key == 'H':
                newH = kwargs['H']
                continue
            if key == 'w':
                new_w = kwargs['w']
                continue

        self.pencil.set( M=newM, N=newN )
        self.symPencil.set( M=0.5*(newM+newM.T), N=0.5*(newN+newN.T) )

        self.H = newH
        self.w = new_w
        self._compute_polys()

    def get_polys(self):
        '''
        Returns: - numerator polynomial n(λ)
                 - denominator polynomial d(λ) 
        '''
        return self.n_poly, self.d_poly
    
    def v(self, l):
        '''
        Compute v(λ) using polynomials
        '''
        return np.array([ v_elem_poly(l) / self.det_poly(l) for v_elem_poly in self.v_poly ])

    def equilibria(self) -> list[dict]:
        ''' 
        Boundary equilibrium solutions can be computed from 
        the q-function by solving q(λ) = n(λ)/d(λ) = 1, or the roots of n(λ) - d(λ).

        Returns: a list of dictionaries with all boundary equilibrium solutions 
        and their stability numbers (if dim == 2, otherwise stability is None).
        '''
        zeros = self.zero_poly.roots()
        real_zeros = np.array([ z.real for z in zeros if np.abs(z.imag) < self.real_zero_tol and z.real >= 0.0 ])
        real_zeros.sort()

        sols = [ {"lambda": z} for z in real_zeros ]
        for sol in sols:  
            sol["stability"] = max( self.stability(sol["lambda"]) )
        return sols

    def stability(self, l):
        '''
        Returns the eigenvalues of the stability matrix S computed at λ
        '''
        P = self.pencil(l)
        grad_h = self.H @ self.v(l)
        Proj = np.eye(self.dim) - (np.outer(grad_h, grad_h) / np.linalg.norm(grad_h)**2)
        S = Proj @ (P + P.T) @ Proj
        stabilityEigs = np.array([ eig for eig in np.linalg.eigvals(S) if np.abs(eig) > self.stability_zero_tol ])

        return stabilityEigs

    def orthogonal_nullspace(self) -> MultiPoly:
        ''' 
        Computes the matrix polynomial O(λ) orthogonal to H N(λ),
        where N(λ) is the pencil minimum nullspace polynomial.
        '''
        nullspace_poly = self.pencil.nullspace()
        return poly_nullspace( nullspace_poly.T @ self.H )

    def regular_pencil(self):
        '''
        Computes the regular matrix pencil P(λ) O(λ).T, 
        where O(λ) is the matrix polynomial orthogonal to the nullspace polynomial Λ(λ) Z.

        OBS: useful for general polynomial kernels, resulting in singular pencils.
        '''
        if self.pencil.type == "regular":
            return self.pencil
        
        # Get slice of polynomial that is orthogonal to the nullspace 
        O_poly = self.orthogonal_nullspace()
        Or_poly = O_poly[0:self.pencil.shape[0], :]

        return self.pencil.to_poly_array() @ Or_poly.T

    def plot(self, ax: Axes, res: float = 0.0, q_limits=(-10, 500)):
        '''
        Plots the Q-function for analysis.
        '''
        q_min, q_max = q_limits[0], q_limits[1]
        lambdaRange = []

        ''' Add real eigenvalues to range '''
        realEigens = self.pencil.get_real_eigen()
        for eig in realEigens: 
            if np.abs(eig.eigenvalue.real) < np.inf and np.abs(eig.eigenvalue.real) > -np.inf:
                lambdaRange.append(eig.eigenvalue.real)

        ''' Add real eigenvalues to range '''
        has_real_spectra = self.pencil.has_real_spectra()
        if has_real_spectra:
            stabilityEigens = self.symmetricPencil.get_real_eigen()
            for eig in stabilityEigens:
                if np.abs(eig.eigenvalue.real) < np.inf and np.abs(eig.eigenvalue.real) > -np.inf: 
                    lambdaRange.append(eig.eigenvalue.real)

        ''' Add equilibrium solutions to range '''
        sols = self.equilibria()
        for sol in sols: 
            lambdaRange.append(sol["lambda"])

        ''' Using range min, max values, generate λ range to be plotted '''
        factor = 10
        l_min, l_max = -100,  100
        if lambdaRange: l_min, l_max = min(lambdaRange), max(lambdaRange)

        deltaLambda = l_max - l_min
        l_min -= deltaLambda/factor
        l_max += deltaLambda/factor

        if res == 0.0: res = deltaLambda/20000

        lambdas = np.arange(l_min, l_max, res)

        ''' Loops through each λ on range to find stable/unstable intervals '''
        is_inserting_nsd, is_inserting_psd = False, False
        nsd_intervals, psd_intervals = [], []
        for l in lambdas:

            eigS = self.stability(l)

            ''' Negative definite (stability) intervals '''
            if np.all( eigS < 0.0 ):
                if not is_inserting_nsd:
                    nsd_intervals.append([ l, np.inf ])
                is_inserting_nsd = True
            elif is_inserting_nsd:
                nsd_intervals[-1][1] = l
                is_inserting_nsd = False  

            ''' Non-negative definite (instability) intervals '''
            if np.any( eigS > 0.0 ):
                if not is_inserting_psd:
                    psd_intervals.append([ l, np.inf ])
                is_inserting_psd = True
            elif is_inserting_psd:
                psd_intervals[-1][1] = l
                is_inserting_psd = False  

        strip_size = (q_max - q_min)/40
        strip_alpha = 0.6

        ''' Plot nsd strips '''
        for interval in nsd_intervals:
            xy = (interval[0], -strip_size/2)
            if interval[1] == np.inf: interval[1] = l_max
            length = interval[1] - interval[0]
            rect = patches.Rectangle(xy, length, strip_size, facecolor=mcolors.TABLEAU_COLORS['tab:orange'], alpha=strip_alpha)
            ax.add_patch(rect)

        ''' Plot psd strips '''
        for interval in psd_intervals:
            xy = (interval[0], -strip_size/2)
            if interval[1] == np.inf: interval[1] = l_max
            length = interval[1] - interval[0]
            rect = patches.Rectangle(xy, length, strip_size, facecolor='cyan', alpha=strip_alpha)
            ax.add_patch(rect)

        print(f"Stability intervals = {nsd_intervals}")
        print(f"Instability intervals = {psd_intervals}")

        ''' Plot zero lines '''
        ax.plot( [l_min, l_max], [ 0.0, 0.0 ], 'k' )       # horizontal axis
        ax.plot( [ 0.0, 0.0 ], [ q_min, q_max ], 'k' )          # vertical axis

        ''' Plot the solution line q(λ) = 1 '''
        ax.plot( lambdas, [ 1.0 for l in lambdas ], 'r--' )

        ''' Plot the Q-function q(λ) '''
        q_array = [ self.n_poly(l) / self.d_poly(l) for l in lambdas ]
        ax.plot( lambdas, q_array )

        ''' Plots equilibrium (stable/unstable) λ solutions satisfying q(λ) = 1 '''
        for k, sol in enumerate(sols):
            stability = sol["stability"]
            label_txt = f"{k+1} stability = {stability:2.2f}"
            ax.text(sol["lambda"], 1.0, f"{k+1}", color='k', fontsize=12)
            if stability > 0:
                ax.plot( sol["lambda"], 1.0, 'bo', label=label_txt) # unstable solutions
            if stability < 0:
                ax.plot( sol["lambda"], 1.0, 'ro', label=label_txt ) # stable solutions

        ''' Plots arrows from eigenvalues of equivalent symmetric pencil '''
        if has_real_spectra:
            for k, eig in enumerate(stabilityEigens):
                arrowLen = q_max/10
                arrowWidth = (l_max - l_min)/100
                if eig.inertia > 0.0:
                    direction = q_max/8
                else:
                    direction = - q_max/8
                ax.arrow(eig.eigenvalue, 0.0, 0.0, direction, edgecolor='green', facecolor='green', head_length=arrowLen, head_width=arrowWidth)

        ''' Sets axes limits and legends '''
        ax.set_xlim(l_min, l_max)
        ax.set_ylim(q_min, q_max)
        ax.legend()

    def compatibilize(self, plant: DynamicSystem, clf_dict: dict, cbf_dict, p = 1.0):
        '''
        This method computes a compatible Hessian matrix for the CLF,
        It recomputes the pencil and its q-function many times.

        clf_dict: - dict containing the function for computing the hessian matrix Hv, its degrees of freedom and its center
        cbf_dict: - dict containing the CBF hessian matrix Hh and its center

        '''
        if isinstance(plant, LinearSystem):
            A, B = plant._A, plant._B
        else:
            raise NotImplementedError("Currently, compatibilization is implemented linear systems only.")

        # Stores pencil variables for latter restauration
        auxM, auxN = self.pencil.M, self.pencil.N
        old_n_poly, old_d_poly = self.n_poly, self.d_poly
        oldZ, oldQ = self.Z, self.Q
        old_computed_from = self.computed_from

        ''' ------------------------------ Compatibilization code ------------------------------ '''
        Hvfun = clf_dict["Hv_fun"]      # function for computing Hv
        x0 = clf_dict["center"]         # clf center
        Hv0 = clf_dict["Hv"]             # initial Hv

        n = self.shape[0]
        ndof = int(n*(n+1)/2)

        Hh = cbf_dict["Hh"]             # cbf Hessian
        xi = cbf_dict["center"]         # cbf center

        M = B @ B.T @ Hh

        def cost(var: np.ndarray):
            ''' Objective function: find closest compatible CLF '''
            return np.linalg.norm( Hvfun(var) - Hv0 )

        def psd_constr(var: np.ndarray):
            ''' Returns the eigenvalues of R matrix. '''
            N = B @ B.T @ Hvfun(var) - A

            # Recompute pencil with new values
            self.__init__(M, N)

            # Recompute q-function
            self.inverse()
            self.qfunction(H = Hh, w = N @ (xi - x0) )

            if self.n_poly.degree() < self.d_poly.degree():

                # NON-SINGULAR CASE
                leading_c = self.zero_poly.coef[-1]
                zeros = self.zero_poly.roots()
                leading_c = self.zero_poly.coef[-1]
                zeros = self.zero_poly.roots()

                real_zeros = [ z.real for z in zeros if np.abs(z.imag) < self.real_zero_tol ]
                real_zeros = [ z.real for z in zeros if np.abs(z.imag) < self.real_zero_tol ]
                min_zero, max_zero = min(real_zeros), max(real_zeros)

                psd_poly = Poly([1.0])
                for z in zeros:
                    if z == min_zero or z == max_zero: continue
                    if z == min_zero or z == max_zero: continue
                    psd_poly *= Poly([-z, 1.0])

                psd_poly = MultiPoly.from_nppoly( psd_poly )
            else:
                # SINGULAR CASE
                psd_poly = MultiPoly.from_nppoly( self.zero_poly )
                psd_poly = MultiPoly.from_nppoly( self.zero_poly )

            psd_poly.sos_decomposition()
            R = psd_poly.sos_matrix

            tol = 0.01
            eigs = np.linalg.eigvals(R - tol*np.eye(R.shape[0]))
            return eigs

        def num_constr(var: np.ndarray):
            ''' Returns the error between the n(λ) polynomials '''
            N = B @ B.T @ Hvfun(var) - A
            self.__init__(M,N)
            self.qfunction(H = Hh, w = N @ (xi - x0) )
            return np.linalg.norm( old_n_poly.coef - self.n_poly.coef )

        def den_constr(var: np.ndarray):
            ''' Returns the error between the d(λ) polynomials '''
            N = B @ B.T @ Hvfun(var) - A
            self.__init__(M,N)
            self.qfunction(H = Hh, w = N @ (xi - x0) )
            return np.linalg.norm( old_d_poly.coef - self.d_poly.coef )

        def const_eigenvec(var: np.ndarray):
            ''' Constraint for constant pencil eigenvectors '''
            N = B @ B.T @ Hvfun(var) - A
            self.__init__(M,N)
            self.eigen()
            distZ = np.linalg.norm(oldZ.T @ self.Z - np.eye(n), 'fro')
            distQ = np.linalg.norm(oldQ.T @ self.Q - np.eye(n), 'fro')
            return distZ + distQ

        constr = [ {"type": "ineq", "fun": psd_constr} ]
        # constr += [ {"type": "eq", "fun": const_eigenvec} ]

        if "Hv" in clf_dict.keys(): objective = cost
        else: objective = lambda var: 0.0

        var0 = np.random.randn(ndof)
        if "Hv" in clf_dict.keys(): objective = cost
        else: objective = lambda var: 0.0

        var0 = np.random.randn(ndof)
        sol = minimize( objective, var0, constraints=constr, options={"disp": True, "maxiter": 1000} )
        Hv = Hvfun(sol.x)

        # Restores pencil variables
        # self.M, self.N = auxM, auxN
        # self.n_poly, self.d_poly = old_n_poly, old_d_poly
        # self.computed_from = old_computed_from

        return Hv

    def __call__(self, l):
        ''' Calling method '''
        return self.n_poly(l) / self.d_poly(l)

class CLFCBFPair():
    '''
    Class for a CLF-CBF pair. Computes the q-function, equilibrium points and critical points of the q-function.
    '''
    def __init__(self, clf, cbf):

        self.eigen_threshold = 0.000001
        self.update(clf = clf, cbf = cbf)

    def update(self, **kwargs):
        '''
        Updates the CLF-CBF pair.
        '''
        for key in kwargs:
            if key == "clf":
                self.clf = kwargs[key]
            if key == "cbf":
                self.cbf = kwargs[key]

        self.Hv = self.clf.get_hessian()
        self.x0 = self.clf.get_critical()
        self.Hh = self.cbf.get_hessian()
        self.p0 = self.cbf.get_critical()
        self.v0 = self.Hv @ ( self.p0 - self.x0 )

        self.pencil = LinearMatrixPencil( self.cbf.get_hessian(), self.clf.get_hessian() )
        self.dim = self.pencil.dim

        self.compute_q()
        self.compute_equilibrium()
        # self.compute_equilibrium2()
        self.compute_critical()

    def compute_equilibrium2(self):
        '''
        Compute the equilibrium points using new method.
        '''
        temp_P = -(self.Hv @ self.x0).reshape(self.dim,1)
        P_matrix = np.block([ [ self.Hv  , temp_P                        ],
                              [ temp_P.T , self.x0 @ self.Hv @ self.x0 ] ])

        temp_Q = -(self.Hh @ self.p0).reshape(self.dim,1)
        Q_matrix = np.block([ [ self.Hh  , temp_Q                        ],
                              [ temp_Q.T , self.p0 @ self.Hh @ self.p0 ] ])

        pencil = LinearMatrixPencil( Q_matrix, P_matrix )
        # print("Eig = " + str(pencil.eigenvectors))

        # self.equilibrium_points2 = np.zeros([self.dim, self.dim+1])1
        self.equilibrium_points2 = []
        for k in range(np.shape(pencil.eigenvectors)[1]):
            # if np.abs(pencil.eigenvectors[-1,k]) > 0.0001:
            # print(pencil.eigenvectors)
            self.equilibrium_points2.append( (pencil.eigenvectors[0:-1,k]/pencil.eigenvectors[-1,k]).tolist() )

        self.equilibrium_points2 = np.array(self.equilibrium_points2).T

        # print("Lambda 1 = " + str(pencil.lambda1))
        # print("Lambda 2 = " + str(pencil.lambda2))
        # print("Eq = " + str(self.equilibrium_points2))

    def compute_q(self):
        '''
        This method computes the q-function for the pair.
        '''
        # Compute denominator of q
        pencil_eig = self.pencil.eigenvalues
        pencil_char = self.pencil.characteristic_poly
        den_poly = np.polynomial.polynomial.polymul(pencil_char, pencil_char)

        detHv = np.linalg.det(self.Hv)
        try:
            Hv_inv = np.linalg.inv(self.Hv)
            Hv_adj = detHv*Hv_inv
        except np.linalg.LinAlgError as error:
            print(error)
            return

        # This computes the pencil adjugate expansion and the set of numerator vectors by the adapted Faddeev-LeVerrier algorithm.
        D = np.zeros([self.dim, self.dim, self.dim])
        D[:][:][0] = pow(-1,self.dim-1) * Hv_adj

        Omega = np.zeros( [ self.dim, self.dim ] )
        Omega[0,:] = D[:][:][0].dot(self.v0)
        for k in range(1,self.dim):
            D[:][:][k] = np.matmul( Hv_inv, np.matmul(self.Hh, D[:][:][k-1]) - pencil_char[k]*np.eye(self.dim) )
            Omega[k,:] = D[:][:][k].dot(self.v0)

        # Computes the numerator polynomial
        W = np.zeros( [ self.dim, self.dim ] )
        for i in range(self.dim):
            for j in range(self.dim):
                W[i,j] = np.inner(self.Hh.dot(Omega[i,:]), Omega[j,:])

        num_poly = np.polynomial.polynomial.polyzero
        for k in range(self.dim):
            poly_term = np.polynomial.polynomial.polymul( W[:,k], np.eye(self.dim)[:,k] )
            num_poly = np.polynomial.polynomial.polyadd(num_poly, poly_term)

        residues, poles, k = signal.residue( np.flip(num_poly), np.flip(den_poly), tol=0.001, rtype='avg' )

        index = np.argwhere(np.real(residues) < 0.0000001)
        residues = np.real(np.delete(residues, index))

        # Computes polynomial roots
        fzeros = np.real( np.polynomial.polynomial.polyroots(num_poly) )

        # Filters repeated poles from pencil_eig and numerator_roots
        repeated_poles = []
        for i in range( len(pencil_eig) ):
            for j in range( len(fzeros) ):
                if np.absolute(fzeros[j] - pencil_eig[i]) < self.eigen_threshold:
                    if np.any(repeated_poles == pencil_eig[i]):
                            break
                    else:
                        repeated_poles.append( pencil_eig[i] )
        repeated_poles = np.array( repeated_poles )

        self.q_function = {
                            "denominator": den_poly,
                            "numerator": num_poly,
                            "poles": pencil_eig,
                            "zeros": fzeros,
                            "repeated_poles": repeated_poles,
                            "residues": residues }

    def compute_equilibrium(self):
        '''
        Compute equilibrium solutions and equilibrium points.
        '''
        solution_poly = np.polynomial.polynomial.polysub( self.q_function["numerator"], self.q_function["denominator"] )

        equilibrium_solutions = np.polynomial.polynomial.polyroots(solution_poly)
        equilibrium_solutions = np.real(np.extract( equilibrium_solutions.imag == 0.0, equilibrium_solutions ))
        equilibrium_solutions = np.concatenate((equilibrium_solutions, self.q_function["repeated_poles"]))

        # Extract positive solutions and sort array
        equilibrium_solutions = np.sort( np.extract( equilibrium_solutions > 0, equilibrium_solutions ) )

        # Compute equilibrium points from equilibrium solutions
        self.equilibrium_points = np.zeros([self.dim,len(equilibrium_solutions)])
        for k in range(len(equilibrium_solutions)):
            if all(np.absolute(equilibrium_solutions[k] - self.pencil.eigenvalues) > self.eigen_threshold ):
                self.equilibrium_points[:,k] = self.v_values( equilibrium_solutions[k] ) + self.p0

    def compute_critical(self):
        '''
        Computes critical points of the q-function.
        '''
        dnum_poly = np.polynomial.polynomial.polyder(self.q_function["numerator"])
        dpencil_char = np.polynomial.polynomial.polyder(self.pencil.characteristic_poly)

        poly1 = np.polynomial.polynomial.polymul(dnum_poly, self.pencil.characteristic_poly)
        poly2 = 2*np.polynomial.polynomial.polymul(self.q_function["numerator"], dpencil_char)
        num_df = np.polynomial.polynomial.polysub( poly1, poly2 )

        self.q_critical_points = np.polynomial.polynomial.polyroots(num_df)
        self.q_critical_points = np.real(np.extract( self.q_critical_points.imag == 0.0, self.q_critical_points ))

        # critical_values = self.q_values(self.q_critical)
        # number_critical = len(self.critical_values)

        # # Get positive critical points
        # index, = np.where(self.q_critical > 0)
        # positive_q_critical = self.q_critical[index]
        # positive_critical_values = self.critical_values[index]
        # num_positive_critical = len(self.positive_q_critical)

    def q_values(self, args):
        '''
        Returns the q-function values at given points.
        '''
        numpoints = len(args)
        qvalues = np.zeros(numpoints)
        for k in range(numpoints):
            num_value = np.polynomial.polynomial.polyval( args[k], self.q_function["numerator"] )
            pencil_char_value = np.polynomial.polynomial.polyval( args[k], self.pencil.characteristic_poly )
            qvalues[k] = num_value/(pencil_char_value**2)

        return qvalues

    def v_values( self, lambda_var ):
        '''
        Returns the value of v(lambda) = H(lambda)^{-1} v0
        '''
        pencil_inv = np.linalg.inv( self.pencil.value( lambda_var ) )
        return pencil_inv.dot(self.v0)

class PolynomialCLFCBFPair():
    '''
    Class for polynomial CLF-CBF pairs of the form:
    V(x,P) = m(x) P m(x) and h(x,Q) = m(x) Q m(x) - 1.
    In this initial implementation, the pair is represented by their respective shape matrices P and Q.
    '''
    def __init__(self, P, Q, max_iter = 1000):
        self.update(P = P, Q = Q, max_iter = max_iter)

    def update(self, **kwargs):
        '''
        Updates the CLF-CBF pair.
        '''
        for key in kwargs:
            if key == "P":
                self.P = kwargs[key]
                continue
            if key == "Q":
                self.Q = kwargs[key]
                continue
            if key == "max_iter":
                self.max_iter = kwargs[key]
                continue

        self.pencil = LinearMatrixPencil2( self.Q, self.P )
        self.n = self.pencil.dim-1

        self.C = scipy.linalg.block_diag(np.zeros([self.n,self.n]), 1) # C matrix for PEP

        self.asymptotes = self.compute_asymptotes()
        self.lambdas, self.kappas, self.equilibria, self.initial_lines = self.compute_equilibrium()

    def compute_asymptotes(self):
        '''
        Computes the asymptotes of the graph det( lambda Q - kappa C - P ) = 0.
        Returns a dict whose keys are the angular coefficients of the asymptotes.
        The values for each key are: the associated linear coefficient, in case that angular coefficient (key) is finite;
                                     the associated horizontal position of the asymptote, in case the angular coefficient is +-inf (vertical asymptote).
        '''
        # Compute angular coefficients of the asymptotes
        pencil_angular_coefs = LinearMatrixPencil2(self.Q, self.C)
        angular_coefs = pencil_angular_coefs.eigenvalues
        asymptotes = { angular_coef: [] for angular_coef in angular_coefs }

        # Compute linear coefficients of the asymptotes
        sorted_eigenvalues = np.sort(self.pencil.eigenvalues)
        to_be_deleted = []
        for i in range(len(sorted_eigenvalues)):
            if np.abs(sorted_eigenvalues[i]) == np.inf:
                to_be_deleted.append(i)
        sorted_eigenvalues = np.delete(sorted_eigenvalues, to_be_deleted)
        differences = np.diff(sorted_eigenvalues)
        '''
        Define initializers for the algorithm.
        If +-inf eigenvalues were found, bound the initiliazers to be inside the limits of the finite spectra: important to prevent singularities.
        '''
        initializers = []
        initializers.append( sorted_eigenvalues[0] - differences[0]/2 )
        for k in range(len(differences)):
            initializers.append( sorted_eigenvalues[k] + differences[k]/2 )
        initializers.append( sorted_eigenvalues[-1] + differences[-1]/2 )

        for i in range(len(angular_coefs)):
            if np.abs(angular_coefs[i]) == np.inf:
                null_space_Q = null_space(self.Q).reshape(self.n+1)
                sol = - (null_space_Q @ self.P @ null_space_Q) / (null_space_Q @ self.C @ null_space_Q)
                asymptotes[angular_coefs[i]].append(sol)
                continue
            def compute_trace(s):
                    invPencil = np.linalg.inv(self.pencil.value(s))
                    return np.trace( invPencil @ ( angular_coefs[i]*self.Q - self.C ) )
            for k in range(len(initializers)):
                sols, infodict, ier, mesg = fsolve( compute_trace, initializers[k], factor=0.1, full_output = True )
                if ier == 1:
                    for sol in sols:
                        if np.any( np.abs(asymptotes[angular_coefs[i]] - sol) < 0.00001 ):
                            continue
                        asymptotes[angular_coefs[i]].append(sol)

        return asymptotes

    def compute_equilibrium(self):
        '''
        Computes all equilibrium points of the CLF-CBF pair, using the Parametric Eigenvalue Problem (PEP)
        '''
        # Separate horizontal from non-horizontal asymptotes.
        # Non-horizontal asymptotes are represented by equation \kappa = m \lambda + p
        lambda_positions = []
        non_horizontal_lines = []
        for key in self.asymptotes.keys():
            if np.abs(key) < ZERO_ACCURACY:
                for p in self.asymptotes[key]:
                    if np.any( np.abs(lambda_positions - p) < ZERO_ACCURACY ):
                        continue
                    lambda_positions.append(p)
            else:
                if np.abs(key) == np.inf:
                    non_horizontal = [ ( 0.0,lin_coef ) for lin_coef in self.asymptotes[key] ]
                else:
                    non_horizontal = [ ( 1/key, -lin_coef/key ) for lin_coef in self.asymptotes[key] ]
                non_horizontal_lines = non_horizontal_lines + non_horizontal

        # Compute intersections with non-horizontal asymptotes
        intersection_pts = np.array([], dtype=float).reshape(2,0)
        for k in range(len(lambda_positions)):
            vert_pos = lambda_positions[k]
            for non_horizontal in non_horizontal_lines:
                m, p = non_horizontal[0], non_horizontal[1]
                kappa_val, lambda_val = m*vert_pos + p, vert_pos
                pt = np.array([kappa_val, lambda_val]).reshape(2,1)
                intersection_pts = np.hstack([intersection_pts, pt])
        diffs = np.diff(intersection_pts)
        # diffs = np.hstack([diffs, diffs[:,-1].reshape(2,1)])

        # Compute random initial points
        # def generate_pts(center, R, n):
        #     '''
        #     Generates n random points inside a circle of radius R centered on center,
        #     filtering points with negative y-value.
        #     '''
        #     pts = np.array([], dtype=float).reshape(2,0)
        #     for _ in range(n):
        #         pt = np.random.normal( center, R, 2 )
        #         # if pt[1]>=0:
        #         pts = np.hstack([pts, pt.reshape(2,1)])
        #     return pts

        # Compute random initial points
        # num_internal_points = 10
        # intermediate_pts = np.array([], dtype=float).reshape(2,0)
        # for k in range(np.shape(diffs)[1]):
        #     for i in range(num_internal_points):
        #         d = np.linalg.norm(diffs[:,k])
        #         vert_pt = intersection_pts[:,k] + i*diffs[:,k]/(num_internal_points)
        #         # hor_pt = intersection_pts[:,k] + (-num_internal_points/2 + i)*np.array([d, 0.0])/(num_internal_points)
        #         intermediate_pts = np.hstack([intermediate_pts, vert_pt.reshape(2,1)])

        # Compute intermediary points for defining initial lines
        num_internal_lines = 1
        intermediate_pts = np.array([], dtype=float).reshape(2,0)
        first_pt = intersection_pts[:,0] - diffs[:,0]/2
        intermediate_pts = np.hstack([intermediate_pts, first_pt.reshape(2,1)])
        for k in range(diffs.shape[1]):
            for i in range(1,num_internal_lines+1):
                pt = intersection_pts[:,k] + i*diffs[:,k]/(num_internal_lines+1)
                intermediate_pts = np.hstack([intermediate_pts, pt.reshape(2,1)])
        last_pt = intersection_pts[:,-1] + diffs[:,-1]/2
        intermediate_pts = np.hstack([intermediate_pts, last_pt.reshape(2,1)])

        # Compute the initial lines
        init_lines = []
        for pt in intermediate_pts.T:
            m = -0.1
            p = pt[1] - m*pt[0]
            init_lines.append( { "angular_coef": m, "linear_coef" : p } )

        # Solves the PEP problem for many different initial lines and store non-repeating results
        lambdas, kappas, equilibrium_points = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float).reshape(self.n,0)

        # lambda_p, kappa_p, Z, init_kappas, init_lambdas = solve_PEP( self.Q, self.P, initial_points = intermediate_pts, max_iter = self.max_iter )
        lambda_p, kappa_p, Z, init_kappas, init_lambdas = solve_PEP( self.Q, self.P, initial_lines = init_lines, max_iter = self.max_iter )
        for i in range(len(lambda_p)):
            equal_lambda = np.any( np.abs( lambda_p[i] - lambdas ) < ZERO_ACCURACY )
            # equal_kappa = np.any( np.abs( kappa_p[i] - kappas ) < ZERO_ACCURACY )
            eq = Z[0:-1,i].reshape(self.n,1)
            equal_eigenvec = np.any( np.linalg.norm( eq - equilibrium_points, axis=0 ) < ZERO_ACCURACY )
            if equal_lambda and equal_eigenvec:
                continue
            lambdas = np.hstack([lambdas, lambda_p[i]])
            kappas = np.hstack([kappas, kappa_p[i]])
            equilibrium_points = np.hstack([equilibrium_points, eq])

        # init_pts = np.vstack([init_kappas, init_lambdas])

        return lambdas, kappas, equilibrium_points, init_lines