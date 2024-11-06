import scipy as sp
import numpy as np
import warnings 

from scipy import signal
from scipy.optimize import fsolve, minimize
from scipy.linalg import null_space, inv
from dataclasses import dataclass

from numpy.polynomial import polynomial as poly
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial.polynomial import polydiv, polymul

from common import vector2sym, sym2vector
from dynamic_systems import DynamicSystem, LinearSystem
from functions import MultiPoly

ZERO_ACCURACY = 1e-9

@dataclass
class Eigen():
    alpha: complex
    beta: float
    eigenvalue: complex
    l_eigenvectors: list | np.ndarray
    r_eigenvectors: list | np.ndarray

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
    if residue > 1e-12: 
        warnings.warn(f"Large residue detected on linear system solution = {residue}")

    if max_deg == 0: max_deg = 1
    x_poly = np.array([[ Poly([0.0 for _ in range(max_deg) ]) for j in range(r) ] for i in range(n) ])
    for (i,j), c in np.ndenumerate(x_coefs):
        exp = int(i/n)
        x_poly[i%n,j].coef[exp] = c

    return x_poly

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

class MatrixPencil():
    '''
    Class for Linear Matrix Pencils of the form P(λ) = λ M - N.
    Computes generalized eigenvalues/eigenvector pairs, rational q-functions, among others.
    '''
    def __init__(self, M: list | np.ndarray, N: list | np.ndarray):

        if isinstance(M, list): M = np.array(M)
        if isinstance(N, list): N = np.array(N)
        
        # Validate passed parameters.
        if M.ndim != 2 or N.ndim != 2:
            raise TypeError("M and N must be two dimensional arrays.")

        if M.shape != N.shape:
            raise TypeError("Matrix dimensions are not equal.")

        if M.shape[0] == M.shape[1]: 
            self.type = 'regular'
        else: 
            self.type = 'singular'

        self.M, self.N = M, N
        self.shape = M.shape
        self.poly = MultiPoly(kernel=[(0,), (1,)], coeffs=[ -self.N, self.M ])
        self.eigens = None

        self.max_order = 100
        self.nullspace_poly = None

        self.trim_tol = 1e-8
        self.real_zero_tol = 1e-6

        self.trim_tol = 1e-8
        self.real_zero_tol = 1e-6

        self.n_poly = None
        self.d_poly = None


        self.computed_from = {"H": None, "w": None}

        self.needs_update = {"eigen": True, 
                             "nullspace": True,
                             "inverse": True,
                              }

    def set(self, M, N):
        '''
        Update method for the pencil. If all is set, update everything 
        '''
        self.__init__(M, N)
        for key in self.needs_update.keys(): self.needs_update[key] = True

    def eigen(self):
        '''
        Computes the generalized eigenvalues/eigenvectors of the pencil using the QZ decomposition.
        It simultaneously decomposes the two matrices of P(λ) = λ M - N as M = Q MM Z' and N = Q NN Z',
        where MM is block upper triangular, NN is upper triangular and Q, Z are unitary matrices.

        If the pencil is singular, compute the nullspace first and then 
        '''
        if not self.needs_update["eigen"]: return
        if self.type == 'singular':
            raise NotImplementedError("Eigenvalue computation (currently) not implemented for singular matrix pencils.")

        # Compute the pencil eigenvalues
        self.MM, self.NN, beta, alpha, self.Q, self.Z = sp.linalg.ordqz(self.M, self.N, output='real')

        self.eigens = []
        for k in range(len(alpha)):
            if beta[k] != 0:
                self.eigens.append( Eigen(alpha[k], beta[k], alpha[k]/beta[k], [], []) )
            else:
                self.eigens.append( Eigen(alpha[k], 0.0, np.inf, [], []) )

        # Compute the pencil eigenvectors
        for eig in self.eigens:

            Qright = null_space( self(eig.alpha, eig.beta) )
            Qleft = null_space( self(eig.alpha, eig.beta).T )

            eig.r_eigenvectors.append( Qright )
            eig.l_eigenvectors.append( Qleft )

        self.needs_update["eigen"] = False

    def nullspace(self):
        ''' 
        Computes the nullspace polynomial Λ(λ) Z of the singular pencil λ M - N, satisfying
        (λ M - N) Λ(λ) Z = 0 identically for all λ
        '''
        if not self.needs_update["nullspace"]: return

        # General singular case
        n, p = self.shape[0], self.shape[1]
        for deg in range(0, self.max_order):

            N = np.zeros([ (deg+2)*n, (deg+1)*p ])
            for i in range(deg+1):
                N[ i*n:(i+1)*n , i*p:(i+1)*p ] = - self.N
                N[ (i+1)*n:(i+2)*n , i*p:(i+1)*p ] = + self.M

            Null = sp.linalg.null_space(N)

            # Regular case
            if n == p and Null.size == 0:
                self.nullspace_poly = MultiPoly( kernel=[(0,)], coeffs=[np.zeros((p,1))] )
                return 

            # End case
            if Null.size != 0:
                break
        
        # Extract matrices
        coefs = [ Null[d*p:(d+1)*p] for d in range(0, deg+1) ]

        self.nullspace_poly = MultiPoly(kernel=[(k,) for k in range(0,deg+1)], coeffs=coefs)
        self.needs_update["nullspace"] = False

    def get_poly(self):
        ''' Returns matrix polynomial equivalent to the pencil '''
        return self.poly

    def get_real_eigen(self):
        ''' Returns the sorted real pencil eigenvalues '''

        self.eigen()

        tol = 1e-12
        real_eigens = []
        for eig in self.eigens:
            if np.abs(eig.eigenvalue.imag) < tol:
                real_eigens.append(eig.eigenvalue.real)
        
        real_eigens = np.array(real_eigens)
        real_eigens.sort()

        return real_eigens

    def get_nullspace(self):
        '''  Returns the nullspace polynomial'''

        self.nullspace()
        return self.nullspace_poly

    def get_qfunction(self):
        '''
        Returns the numerator and denominator polynomials n(λ) and d(λ) 
        of the q-function, if already computed.
        '''
        if not hasattr(self, "n_poly") or not hasattr(self, "d_poly"):
            raise Exception("Q-function was not yet computed.")

        return self.n_poly, self.d_poly

    def inverse(self):
        '''
        Computes the pencil adjoint polynomial matrix adj(λ) and polynomial determinant det(λ).
        The pencil inverse is then given by P^(-1)(λ) = 1/det(λ) adj(λ).
        '''
        if not self.needs_update["inverse"]: 
            return

        n = self.M.shape[0]

        ''' Computes the pencil determinant polynomial expression '''
        blk_dims, blk_poles, blk_adjs = [], [], []
        for i in range(n):

            # 2X2 BLOCKS OF COMPLEX CONJUGATE PENCIL EIGENVALUES
            if i < n-1 and self.MM[i+1,i] != 0.0:
                blk_dims.append(2)

                MMblock = self.MM[i:i+2,i:i+2]
                NNblock = self.NN[i:i+2,i:i+2]

                a = np.linalg.det(MMblock)
                b = -( MMblock[0,0] * NNblock[1,1] + NNblock[0,0] * MMblock[1,1] )
                c = np.linalg.det(NNblock)
                blk_poles.append( Poly([ c, b, a ]) )

                adj11 = Poly([ -NNblock[1,1],  MMblock[1,1] ])
                adj12 = Poly([           0.0, -MMblock[0,1] ])
                adj21 = Poly([           0.0, -MMblock[1,0] ])
                adj22 = Poly([ -NNblock[0,0],  MMblock[0,0] ])
                blk_adjs.append( np.array([[ adj11, adj12 ],[ adj21, adj22 ]]) )

            # 1X1 BLOCKS OF REAL PENCIL EIGENVALUES
            else:
                blk_dims.append(1)

                MMblock = self.MM[i,i]
                NNblock = self.NN[i,i]

                blk_poles.append( Poly([ -NNblock, MMblock ]) )
                blk_adjs.append( np.array([Poly(1.0)]) )
        
        self.determinant = np.prod([ pole for pole in blk_poles ])

        ''' Computes the pencil adjoint matrix '''
        num_blks = len(blk_dims)
        self.adjoint = np.array([[ Poly([0.0]) for _ in range(n) ] for _ in range(n) ])
        for i in range(num_blks-1, -1, -1):
            blk_i_slice = slice( i*blk_dims[i], (i+1)*blk_dims[i] )

            for j in range(i, num_blks):
                blk_j_slice = slice( j*blk_dims[j], (j+1)*blk_dims[j] )

                # Computes ADJOINT DIAGONAL BLOCKS
                if j == i:
                    poles_ij = np.array([[ np.prod([ pole for k, pole in enumerate(blk_poles) if k != i ]) ]])
                    Lij = poles_ij * blk_adjs[j]

                # Computes ADJOINT UPPER TRIANGULAR BLOCKS
                else:
                    Tii = self.MM[ blk_i_slice, blk_i_slice ]
                    Sii = self.NN[ blk_i_slice, blk_i_slice ]

                    b_poly = np.array([[ Poly([0.0]) for _ in range(blk_dims[j]) ] for _ in range(blk_dims[i]) ])
                    for k in range(i+1, j+1):
                        blk_k_slice = slice( k*blk_dims[k], (k+1)*blk_dims[k] )

                        # Compute polynomial (λ Tik - Sik) and get the kj slice of adjoint
                        Tik = self.MM[ blk_i_slice, blk_k_slice ]
                        Sik = self.NN[ blk_i_slice, blk_k_slice ]
                        poly_ik = np.array([[ Poly([ -Sik[a,b], Tik[a,b] ]) for b in range(Tik.shape[1]) ] for a in range(Tik.shape[0]) ])
                        adjoint_kj = self.adjoint[ blk_k_slice, blk_j_slice ]

                        b_poly -= poly_ik @ adjoint_kj

                    Lij = solve_poly_linearsys( Tii, Sii, b_poly )

                # Populate adjoint matrix
                self.adjoint[ blk_i_slice, blk_j_slice ] = Lij

        self.needs_update["inverse"] = False

    def qfunction(self, 
                  H: list | np.ndarray = None, 
                  w: list | np.ndarray = None):
        '''
        Computes the numerator and denominator numpy polynomials n(λ), d(λ) 
        of the q-function q(λ) = n(λ)/d(λ) from the previously computed pencil adjoint and determinant.
        In summary, the q-function is the solution to the following problem:
        - P(λ) v(λ) = w, 
        - q(λ) = v(λ).T @ Hh v(λ).

        Inputs: - matrix Hh from q(λ) = v(λ).T @ Hh v(λ)
                - vector w from P(λ) v(λ) = w
        '''
        if H is None and self.computed_from["H"] is None:
            raise TypeError("H must be passed to compute the q-function.")

        if w is None and self.computed_from["w"] is None:
            raise TypeError("w must be passed to compute the q-function.")

        if isinstance(H, list): H = np.array(H)
        if isinstance(w, list): w = np.array(w)

        no_need_recompute = np.all(H == self.computed_from["H"])
        no_need_recompute = no_need_recompute and np.all(w == self.computed_from["w"])
        if no_need_recompute: return

        # Validate passed parameters.
        n, p = self.shape[0], self.shape[1]
        if H.shape != (p,p):
            raise TypeError("H must have compatible dimensions.")

        if len(w) != n:
            raise TypeError("Vector w must have compatible dimensions.")

        self.computed_from = {"H": H, "w": w}

        if self.needs_update["eigen"]: self.eigen()

        barHh = self.Z.T @ H @ self.Z
        barw = self.Q.T @ w

        if self.needs_update["inverse"]: self.inverse()

        self.n_poly = ( barw.T @ self.adjoint.T @ barHh @ self.adjoint @ barw ).trim(tol=self.trim_tol)
        self.d_poly = ( self.determinant**2 ).trim(tol=self.trim_tol)
        
        self.zero_poly = ( self.n_poly - self.d_poly ).trim(tol=self.trim_tol)

    def qfunction_value(self, l: float, H: list | np.ndarray, w: list | np.ndarray) -> float:
        '''
        Computes the q-function value q(λ) = n(λ)/d(λ) for a given λ (for DEBUG only)
        '''
        P = self(l)
        v = inv(P) @ w
        return v.T @ H @ v 

    def equilibria(self) -> list[dict]:
        ''' 
        Boundary equilibrium solutions can be computed from 
        the q-function by solving q(λ) = n(λ)/d(λ) = 1, or the roots of n(λ) - d(λ).

        Returns: a list of dictionaries with all boundary equilibrium solutions 
        and their stability numbers (if dim == 2, otherwise stability is None).
        '''
        H = self.computed_from["H"]
        w = self.computed_from["w"]

        zeros = self.zero_poly.roots()
        real_zeros = np.array([ z.real for z in zeros if np.abs(z.imag) < self.real_zero_tol ])
        real_zeros.sort()

        sols = [ {"lambda": z} for z in real_zeros ]
        for sol in sols:
            if self.shape == (2,2):
                l = sol["lambda"]
                P = self(l)
                v_polys = self.adjoint @ w
                grad_h = H @ np.array([ v(l) for v in v_polys ])
                perp = np.array([ grad_h[1], -grad_h[0] ])
                sol["stability"] = perp.T @ P @ perp
            else:
                sol["stability"] = None

        return sols

    def orthogonal_nullspace(self, H: list | np.ndarray):
        ''' 
        Computes the matrix polynomial O(λ) orthogonal to the nullspace polynomial Λ(λ) Z,
        that is, O(λ) H Λ(λ) Z = 0 
        '''
        if isinstance(H, list): H = np.array(H)

        n, p = self.shape[0], self.shape[1]
        if H.shape != (p, p): 
            raise TypeError(f"Passed H must be ({p},{p})")

        self.nullspace()
        zdim = self.nullspace_poly.shape[1]
        Zcoefs = self.nullspace_poly.coeffs

        numZcoefs = len(self.nullspace_poly.coeffs)

        for deg in range(0, self.max_order):

            Coef = np.zeros(( (deg+1)*p, numZcoefs*zdim ))

            for i in range(0, numZcoefs-deg):
                aux = H @ np.hstack([ Zcoefs[k] for k in range(0, numZcoefs - i) ])
                Coef[ i*p:(i+1)*p, : ] = np.hstack([ np.zeros((p,zdim*i)), aux ])

            Null = sp.linalg.null_space( Coef.T )
            if Null.size != 0:
                break

        M = Null.T
        Mcoefs = [ M[:, i*p:(i+1)*(p) ] for i in range(deg+1) ]

        return MultiPoly(kernel=[(k,) for k in range(0, deg+1) ], coeffs=Mcoefs)

    def regular_pencil(self, H: list | np.ndarray):
        '''
        Using H of appropriate size, return regular matrix pencil P(λ) O(λ).T
        with O(λ) H Λ(λ) Z = 0
        '''
        if self.type == "regular":
            return self
        
        # Get slice of polynomial that is orthogonal to the nullspace 
        O_poly = self.orthogonal_nullspace(H)
        Or_poly = O_poly[0:self.shape[0], :]

        return self.poly @ Or_poly.T

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
        auxM, auxN = self.M, self.N
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

    def plot_qfunction(self, ax, res: float = None, q_limits=(0, 100)):
        ''' Plot q(λ) at axes ax.'''

        q_min, q_max = q_limits[0], q_limits[1]

        lambdas = []
        sols = self.equilibria()
        for sol in sols:
            lambdas.append(sol["lambda"])

        factor = 1.5
        if lambdas:
            l_min, l_max = factor*min(lambdas), factor*max(lambdas)
        else:
            l_min, l_max = 0.0, 1000

        if res == None: 
            res = (l_max - l_min)/1000
        lambdas = np.arange(l_min, l_max, res)

        # Plot real eigenvalues
        for eig in self.get_real_eigen():
            ax.plot( [ eig for _ in lambdas ], np.linspace(q_min, q_max, len(lambdas)), 'b--' )

        # Plot the solution line
        ax.plot( lambdas, [ 1.0 for l in lambdas ], 'r--' )

        # Plot q-function
        q_array = [ self.n_poly(l) / self.d_poly(l) for l in lambdas ]
        ax.plot( lambdas, q_array, label='q' )

        # Plots boundary equilibrium solutions
        for sol in sols:
            if sol["stability"] > 0: ax.plot( sol["lambda"], 1.0, 'bo' ) # stable solutions
            if sol["stability"] < 0: ax.plot( sol["lambda"], 1.0, 'ro' ) # unstable solutions

        ax.set_xlim(l_min, l_max) 
        ax.set_ylim(q_min, q_max)
        ax.legend()

    def __call__(self, l: int | float, beta: int | float = 1.0) -> np.ndarray:
        '''
        Returns pencil value.
        If only one argument is passed, it is interpreted as the λ value and method returns matrix P(λ) = λ M - N.
        If two arguments are passed, they are interpreted as α, β values and method returns P(α, β) = α M - β N.
        '''
        return l * self.M  - beta * self.N

    def __str__(self) -> str:
        '''
        Print the pencil P(λ) = λ M - N.
        '''
        np.set_printoptions(precision=3, suppress=True)
        ret_str = '{}'.format(type(self).__name__) + " = \u03BB M - N \n"
        ret_str = ret_str + 'M = ' + self.M.__str__() + '\n'
        ret_str = ret_str + 'N = ' + self.N.__str__()
        return ret_str

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