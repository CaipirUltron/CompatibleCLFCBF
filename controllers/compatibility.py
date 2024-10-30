import scipy as sp
import numpy as np
import warnings 

from scipy import signal
from scipy.optimize import fsolve, minimize
from scipy.linalg import null_space, inv

from numpy.polynomial import polynomial as poly
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial.polynomial import polydiv, polymul

from common import vector2sym, sym2vector
from dynamic_systems import DynamicSystem, LinearSystem
from functions import MultiPoly

ZERO_ACCURACY = 1e-9

def solve_poly_linearsys(T: np.ndarray, S: np.ndarray, b_poly: np.ndarray) -> np.ndarray:
    '''
    Finds the polynomial array x(λ) that solves (λ T - S) x(λ) = b(λ), where T, S are 1x1 or 2x2
    and b(λ) is a polynomial array of arbitrary order.

    Input: - matrices T, S from linear matrix pencil (λ T - S), where T is 
    '''
    if isinstance(T, (int, float)): T = np.array([[T]])
    if isinstance(S, (int, float)): S = np.array([[S]])

    if T.shape != S.shape:
        raise TypeError("T and S must have the same shape.")
    
    if T.shape[0] != T.shape[0]:
        raise TypeError("T and S must be square matrices.")

    blk_size = T.shape[0]
    bshape = b_poly.shape
    if bshape[0] != blk_size:
        raise TypeError("Number of lines in (λ T - S) and b(λ) must be the same.")

    # Extract arrays from b_poly and store in b_coefs list (variable size)
    bsys = np.zeros((0, bshape[1]))
    for (i,j), poly in np.ndenumerate(b_poly):

        if not isinstance( poly, Poly ):
            raise TypeError("b(λ) is not an array of polynomials.")
        
        # Setup bsys
        b_order = len(poly.coef)
        n_coefs_toadd = b_order - int(bsys.shape[0] / blk_size)
        if n_coefs_toadd > 0:
            bsys = np.vstack([ bsys ] + [ np.zeros((blk_size, bshape[1])) for _ in range(n_coefs_toadd) ])

        for k, c in enumerate(poly.coef):
            bsys[ k * blk_size + i, j ] = c

    # Constructs the Asys and bsys matrices
    b_order = int(bsys.shape[0] / blk_size)
    Asys = np.zeros([ b_order*blk_size, (b_order-1)*blk_size ])
    for i in range(b_order-1):
        Asys[ i*blk_size:(i+1)*blk_size , i*blk_size:(i+1)*blk_size ] = -S
        Asys[ (i+1)*blk_size:(i+2)*blk_size , i*blk_size:(i+1)*blk_size ] = T

    results = np.linalg.lstsq(Asys, bsys, rcond=None)
    x_coefs = results[0]
    residue = np.linalg.norm(results[1])

    if residue > 1e-12: 
        warnings.warn(f"Large residue detected on linear system solution = {residue}")

    x_poly = np.array([[ Poly([0.0 for _ in range(b_order-1) ]) for j in range(bshape[1]) ] for i in range(blk_size) ])
    for (i,j), c in np.ndenumerate(x_coefs):
        exp = int(i/blk_size)
        x_poly[i%blk_size,j].coef[exp] = c

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
            raise TypeError("A and B must be two dimensional arrays.")

        if M.shape != N.shape:
            raise TypeError("Matrix dimensions are not equal.")

        if M.shape[0] != M.shape[1]:
            raise Exception("Matrices are not square.")

        self.M, self.N = M, N
        self.dim = M.shape[0]
        self.eigen()

    def eigen(self):
        '''
        Computes the generalized eigenvalues/eigenvectors of the pencil using the QZ decomposition.
        It simultaneously decomposes the two matrices of P(λ) = λ M - N as M = Q MM Z' and N = Q NN Z',
        where MM is block upper triangular, NN is upper triangular and Q, Z are unitary matrices.
        '''
        # Compute the pencil eigenvalues
        self.MM, self.NN, self.beta, self.alpha, self.Q, self.Z = sp.linalg.ordqz(self.M, self.N, output='real')
        self.eigenvalues = self.alpha / self.beta

        # Compute real eigenvalues 
        self.real_eigenvalues = np.array([ z.real for z in self.eigenvalues if np.abs(z.imag) < 1e-12 ])
        self.real_eigenvalues.sort()

        # Compute the pencil eigenvectors
        self.eigenvectors = []
        for alpha, beta in zip(self.alpha, self.beta):
            Qs = null_space( self(alpha, beta) )
            self.eigenvectors.append( {"eigenvalue": (alpha, beta), "eigenvectors": Qs} )

    def qfunction(self, H: list | np.ndarray, w: list | np.ndarray):
        '''
        Computes the numerator and denominator numpy polynomials n(λ), d(λ) 
        of the q-function q(λ) = n(λ)/d(λ), where:
        - P(λ) v(λ) = w, 
        - q(λ) = v(λ).T @ Hh v(λ).

        Inputs: - matrices M, N from linear matrix pencil P(λ) = λ M - N
                - matrix Hh from q(λ) = v(λ).T @ Hh v(λ)
                - vector w from P(λ) v(λ) = w
        '''
        if isinstance(H, list): H = np.array(H)
        if isinstance(w, list): w = np.array(w)

        # Validate passed parameters.
        if H.shape != (self.dim, self.dim):
            raise TypeError("H must have the same dimension as the pencil matrices.")
        
        n = self.M.shape[0]
        if len(w) != n:
            raise TypeError("Vector w must have the same dimensions as the pencil λ M - N.")

        ''' -------------------- Computes the block dimensions, poles and adjoints (TO BE USED later) ------------------------- '''
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
        
        ''' -------------------------- Computes the adjoint matrix --------------------------------- '''

        num_blks = len(blk_dims)
        adjoint = np.array([[ Poly([0.0]) for _ in range(n) ] for _ in range(n) ])
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
                        Sik = self. NN[ blk_i_slice, blk_k_slice ]
                        poly_ik = np.array([[ Poly([ -Sik[a,b], Tik[a,b] ]) for b in range(Tik.shape[1]) ] for a in range(Tik.shape[0]) ])

                        adjoint_kj = adjoint[ blk_k_slice, blk_j_slice ]

                        b_poly -= poly_ik @ adjoint_kj

                    Lij = solve_poly_linearsys( Tii, Sii, b_poly )

                # Populate adjoint matrix
                adjoint[ blk_i_slice, blk_j_slice ] = Lij

        ''' ---------------- Computes q-function numerator and denominator polynomials ------------------ '''
        Zinv = np.linalg.inv(self.Z)
        barHh = Zinv @ H @ (Zinv.T)
        barw = np.linalg.inv(self.Q) @ w

        self.n_poly = ( barw.T @ adjoint.T @ barHh @ adjoint @ barw )
        self.d_poly = np.prod([ pole**2 for pole in blk_poles ])

        # Trim coefficients
        for k, c in enumerate(self.n_poly.coef):
            if np.abs(c) < 1e-12: 
                self.n_poly.coef[k] = 0.0

        for k, c in enumerate(self.d_poly.coef):
            if np.abs(c) < 1e-12: 
                self.d_poly.coef[k] = 0.0

        self.computed_from = {"H": H, "w": w}

    def qfunction_value(self, l: float, H: list | np.ndarray, w: list | np.ndarray):
        '''
        Computes the q-function value q(λ) = n(λ)/d(λ) for a given λ (for DEBUG only)
        '''
        P = self(l)
        v = inv(P) @ w
        return v.T @ H @ v 

    def get_qfunction(self):
        '''
        Returns the numerator and denominator polynomials n(λ) and d(λ) 
        of the q-function, if already computed.
        '''
        if not hasattr(self, "n_poly") or not hasattr(self, "d_poly"):
            raise Exception("Q-function was not yet computed.")

        return self.n_poly, self.d_poly

    def equilibria(self):
        ''' 
        Assuming the q-function was already computed, 
        boundary equilibrium solutions can be computed from 
        the q-function by solving q(λ) = n(λ)/d(λ) = 1, or the roots of n(λ) - d(λ).

        Returns: a list of dictionaries with all boundary equilibrium solutions 
        and their stability numbers (if dim == 2, otherwise stability is None).
        '''
        if not hasattr(self, "n_poly") or not hasattr(self, "d_poly"):
            raise Exception("Q-function was not yet computed.")

        H = self.computed_from["H"]
        w = self.computed_from["w"]

        zeros = (self.n_poly - self.d_poly).roots()
        real_zeros = np.array([ z.real for z in zeros if np.abs(z.imag) < 1e-12 ])
        real_zeros.sort()

        sols = [ {"lambda": z} for z in real_zeros ]
        for sol in sols:
            if self.dim == 2:
                l = sol["lambda"]
                P = self(l)
                v = np.linalg.inv(P) @ w
                grad_h = H @ v
                perp = np.array([ grad_h[1], -grad_h[0] ])
                sol["stability"] = perp.T @ P @ perp
            else:
                sol["stability"] = None

        return sols

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
        old_computed_from = self.computed_from

        ''' ------------------------------ Compatibilization code ------------------------------ '''
        Hvfun = clf_dict["Hv_fun"]      # function for computing Hv
        x0 = clf_dict["center"]         # clf center
        Hv0 = clf_dict["Hv"]             # initial Hv

        n = self.dim
        ndof = int(n*(n+1)/2)

        Hh = cbf_dict["Hh"]             # cbf Hessian
        xi = cbf_dict["center"]         # cbf center

        M = B @ B.T @ Hh

        def cost(var: np.ndarray):
            '''
            Objective function
            '''
            return np.linalg.norm( Hvfun(var) - Hv0 )

        def psd_constr(var: np.ndarray):
            ''' Returns the eigenvalues of R matrix. '''
            N = B @ B.T @ Hvfun(var) - A

            # Recompute pencil with new values
            self.__init__(M,N)

            # Recompute q-function
            self.qfunction(H = Hh, w = N @ (xi - x0) )
            zero_poly = self.n_poly - self.d_poly

            if self.n_poly.trim(tol=1e-3).degree() < self.d_poly.trim(tol=1e-3).degree():

                # NON-SINGULAR CASE
                zeros = zero_poly.roots()
                real_zeros = [ z.real for z in zeros if np.abs(z.imag) < 1e-3 ]

                min_zero, max_zero = min(real_zeros), max(real_zeros)
                real_remainder = - Poly([-min_zero, 1.0]) * Poly([-max_zero, 1.0])

                # The result of the division by the real remainder is the desired-to-be psd polynomial
                coefs, rem = polydiv(zero_poly.coef, real_remainder.coef)
                rem_norm = np.linalg.norm(rem)
                if rem_norm > 1e-3:
                    warnings.warn(f"Remainder norm is too large = {rem_norm}.")

                psd_poly = MultiPoly.from_nppoly( Poly(coefs.real) )
            else:

                # SINGULAR CASE
                psd_poly = MultiPoly.from_nppoly( zero_poly )

            psd_poly.sos_decomposition()
            R = psd_poly.sos_matrix
            eigsR = np.linalg.eigvals( R )

            tol = 0.1
            return np.linalg.eigvals(R - tol*np.eye(R.shape[0])) 

        def den_constr(var: np.ndarray):
            ''' Returns the error between the d polynomials '''
            N = B @ B.T @ Hvfun(var) - A

            # Recompute pencil with new values
            self.__init__(M,N)

            # Recompute q-function
            self.qfunction(H = Hh, w = N @ (xi - x0) )

            return np.linalg.norm( old_n_poly.coef - self.n_poly.coef )

        constr = [ {"type": "ineq", "fun": psd_constr} ]
        # constr += [ {"type": "eq", "fun": den_constr} ]

        if "Hv" in clf_dict.keys():
            var0 = sym2vector(Hv0)
            objective = cost
        else:
            var0 = np.random.randn(ndof)
            objective = lambda var: 0.0

        sol = minimize( objective, var0, constraints=constr, options={"disp": True, "maxiter": 5000} )
        Hv = Hvfun(sol.x)

        # Restores pencil variables
        # self.M, self.N = auxM, auxN
        # self.n_poly, self.d_poly = old_n_poly, old_d_poly
        # self.computed_from = old_computed_from

        return Hv

    def plot_qfunction(self, ax, res=0.1, q_limits=(0, 100)):
        ''' Plot q(λ) at axes ax.'''

        q_min, q_max = q_limits[0], q_limits[1]

        # Get real eigenvalues
        real_eigs = []
        for eig in self.eigenvalues:
            if np.abs(eig.imag) < 1e-10:
                real_eigs.append(eig.real)
        real_eigs = np.array(real_eigs)
        real_eigs.sort()

        equilibria = []
        sols = self.equilibria()
        for sol in sols:
            equilibria.append(sol["lambda"])

        factor = 1.5
        if equilibria:
            l_min, l_max = factor*min(equilibria), factor*max(equilibria)
        else:
            l_min, l_max = 0.0, 1000

        lambdas = np.arange(l_min, l_max, res)

        # Plot real eigenvalues
        for eig in real_eigs:
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