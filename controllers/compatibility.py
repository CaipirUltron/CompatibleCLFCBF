import math
import numpy as np
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from matplotlib.axes import Axes
from numpy.polynomial import Polynomial as Poly

from scipy import signal
from scipy.optimize import fsolve, minimize
from scipy.linalg import null_space, inv

from common import *
from functions import MultiPoly
from dynamic_systems import DynamicSystem, LinearSystem

class QFunction():
    '''
    General class for Qfunctions q(λ) = v(λ)' H v(λ), where P(λ) v(λ) = w
    and P(λ) is a regular linear matrix pencil.
    '''
    def __init__(self, P: MatrixPencil, H: list | np.ndarray, w: list | np.ndarray):

        if isinstance(H, list): H = np.array(H)
        if isinstance(w, list): w = np.array(w)
        self._verify(P, H, w)

        self.pencil: MatrixPencil = P

        self.H = H
        self.w = w

        ''' Class parameters '''
        self.trim_tol = 1e-8
        self.real_tol = 1e-6
        self.compatibility_params = {"eps1": 1.01,        # eps1 should be > 1
                                     "eps2": 1e-0 }      # eps2 should be small

        ''' Stability/compatibility matrices '''
        stb_dim = self.dim-1
        self.stability_matrix: MatrixPolynomial = MatrixPolynomial.zeros(size=(stb_dim, stb_dim))
        self.stability_pencil: MatrixPencil = None
        self.compatibility_matrix: MatrixPolynomial = MatrixPolynomial.zeros(size=(stb_dim, stb_dim))

        self._compute_polynomials()

    def __call__(self, l):
        ''' Calling method '''
        return self.n_poly(l) / self.d_poly(l)

    def _verify(self, P: MatrixPencil, H: list | np.ndarray, w: list | np.ndarray):
        '''
        Verification method for pencil intialization.
        '''
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

    def _compute_polynomials(self):
        ''' Private method for computing the QFunction polynomials '''

        ''' Computation of Q-function numerator/denominator polynomials '''
        determinant, adjoint = self.pencil.inverse()
        self.n_poly = ( self.w.T @ adjoint.T @ self.H @ adjoint @ self.w )
        self.d_poly = ( determinant**2 )

        # Normalize numerator/denominator coefficients with the mean value of all coefficients
        norm_coef = np.mean([ np.abs(c) for c in self.n_poly.coef ]+[ np.abs(c) for c in self.d_poly.coef ])
        self.n_poly = self.n_poly/norm_coef
        self.d_poly = self.d_poly/norm_coef

        print(f"roots of n(λ) = { self.n_poly.roots() }")
        print(f"roots of d(λ) = { self.d_poly.roots() }")

        ''' Computation of the zero-polynomial, for computing the boundary equilibrium points '''
        self.zero_poly = ( self.n_poly - self.d_poly )

        ''' Computation of stability properties of the boundary equilibrium points '''
        self.divisor_poly, self.v_poly = self._v_poly()

        # print(f"v(λ) = {self.v_poly}")
        # print(f"div(λ) = {self.divisor_poly}")

        self._stability_matrix()

        ''' Computation of compatibility matrix polynomial '''
        self._compatibility_matrix()

        # C = self.compatibility_matrix.sos_decomposition()
        # eigC = np.linalg.eigvals(C)
        # print(f"Compatibility eigenvalues = {eigC}")

    def _v_poly_derivative(self, order=0) -> tuple[Poly, np.ndarray[Poly]]:
        '''
        Computes the derivatives of v(λ) of any order.
        Returns: - scalar divisor polynomial a(λ)
                 - vector polynomial v_poly(λ)
        v(λ) = 1/a(λ) v_poly(λ)
        '''
        det, adjoint = self.pencil.inverse()
        v = math.factorial(order) * adjoint @ self.w

        power = np.eye(self.pencil.shape[0])
        for k in range(order):
            power @= - adjoint @ self.pencil.M

        div = det**(order+1)
        v = power @ v

        return div, v

    def _v_poly(self) -> tuple[Poly, np.ndarray[Poly]]:
        '''
        Compute the v(λ) polynomial
        '''
        return self._v_poly_derivative(order=0)

    def _stability_matrix(self) -> MatrixPolynomial:
        ''' Computes the stability polynomial matrix '''

        nablah = ( self.H @ self.v_poly )

        ''' Computes N(λ), the nullspace matrix polynomial to ∇h(λ) of appropriate degree '''
        S_deg = self.d_poly.degree() - 1
        null_deg = math.floor(S_deg / 2)
        Q_matrix = nullspace( nablah, degree=null_deg )[:,0:self.dim-1]

        ''' Computes the stability matrix polynomial from N(λ) '''
        Psym = self.pencil.symmetric()
        S_matrix = Q_matrix.T @ Psym @ Q_matrix
        self.stability_matrix.update( S_matrix )

        if self.stability_pencil is None:
            self.stability_pencil = self.stability_matrix.companion_form()
        else:
            newM, newN = companion_form( self.stability_matrix.poly_array )
            self.stability_pencil.update( M = newM, N = newN )

    def _compatibility_matrix(self) -> MatrixPolynomial:
        '''
        Compatibility matrix is a sufficient condition for compatibility of the CLF-CBF pair and given Linear System
        '''
        eps1 = self.compatibility_params["eps1"]
        eps2 = self.compatibility_params["eps2"]

        safe_zero_poly = self.n_poly - eps1 * self.d_poly
        identity_part = np.diag([ safe_zero_poly for _ in range(self.dim-1) ])

        symb = self.pencil.symbol
        lambda_poly = np.array([ Poly([0, 1], symbol=symb) ])
        stability_part = eps2 * lambda_poly * self.stability_matrix.poly_array

        compatibility = identity_part + stability_part
        self.compatibility_matrix.update(compatibility)

        # print(f"n(λ) degree = { self.n_poly.degree() }")
        # print(f"d(λ) degree = { self.d_poly.degree() }")
        # print(f"n(λ) - eps1 d(λ) degree = { safe_zero_poly.degree() }")
        # print(f"S(λ) = {self.stability_matrix}")

    def _qvalue(self, l: float, H: list | np.ndarray, w: list | np.ndarray) -> float:
        '''
        Computes the q-function value q(λ) = n(λ)/d(λ) for a given λ (for DEBUG only)
        '''
        P = self.pencil(l)
        v = inv(P) @ self.w
        return v.T @ self.H @ v

    def update(self, **kwargs):
        '''
        QFunction update method.
        Inputs: - M, N - pencil matrices for pencil update
                - H and w - Q-function matrix and constant vector
        '''
        newM, newN = self.pencil.M, self.pencil.N
        for key in kwargs.keys():
            if key == 'M':
                newM = kwargs['M']
                continue
            if key == 'N':
                newN = kwargs['N']
                continue
            if key == 'H':
                self.H = kwargs['H']
                continue
            if key == 'w':
                self.w = kwargs['w']
                continue

        self.pencil.update( M=newM, N=newN )
        self._compute_polynomials()

    def get_polys(self) -> tuple[Poly, Poly]:
        '''
        Returns: - numerator polynomial n(λ)
                 - denominator polynomial d(λ)
        '''
        return self.n_poly, self.d_poly

    def v(self, l):
        '''
        Compute v(λ) using polynomials
        '''
        div = self.divisor_poly(l)
        v = np.array([ poly(l) for _, poly in np.ndenumerate(self.v_poly) ])
        return v / div

    def equilibria(self) -> list[dict]:
        '''
        Boundary equilibrium solutions can be computed from
        the q-function by solving q(λ) = n(λ)/d(λ) = 1, or the roots of n(λ) - d(λ).

        Returns: a list of dictionaries with all boundary equilibrium solutions
        and their stability numbers (if dim == 2, otherwise stability is None).
        '''
        zeros = self.zero_poly.roots()
        real_zeros = np.array([ z.real for z in zeros if np.abs(z.imag) < self.real_tol and z.real >= 0.0 ])
        real_zeros.sort()

        sols = [ {"lambda": z} for z in real_zeros ]
        for sol in sols:
            sol["stability"] = max( self.stability(sol["lambda"]) )
        return sols

    def stability(self, l):
        '''
        Returns the eigenvalues of the stability matrix S computed at λ
        '''
        stabilityEigs = np.array([ eig for eig in np.linalg.eigvals( self.stability_matrix(l) )
                                   if np.abs(eig.imag) < self.real_tol ])
        return stabilityEigs

    def orthogonal_nullspace(self) -> MatrixPolynomial:
        '''
        Computes the matrix polynomial O(λ) orthogonal to H N(λ),
        where N(λ) is the pencil minimum nullspace polynomial.
        '''
        pencil_nullspace = self.pencil.nullspace()
        return nullspace( pencil_nullspace.T @ self.H )

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

        return self.pencil @ Or_poly.T

    def plot(self, ax: Axes, res: float = 0.0, q_limits=(-10, 500)):
        '''
        Plots the Q-function for analysis.
        '''
        q_min, q_max = q_limits[0], q_limits[1]
        lambdaRange = [0.0]

        ''' Add pencil real eigenvalues to range '''
        realEigens = self.pencil.real_eigen()
        for eig in realEigens:
            if np.abs(eig.eigenvalue.real) < np.inf and np.abs(eig.eigenvalue.real) > -np.inf:
                lambdaRange.append(eig.eigenvalue.real)

        ''' Add equilibrium solutions to range '''
        sols = self.equilibria()
        for sol in sols:
            lambdaRange.append(sol["lambda"])

        for eig in self.stability_pencil.real_eigen():
            lambdaRange.append(eig.eigenvalue)

        ''' Using range min, max values, generate λ range to be plotted '''
        factor = 10
        l_min, l_max = -100,  100
        l_min, l_max = min(lambdaRange), max(lambdaRange)

        deltaLambda = l_max - l_min
        l_min -= deltaLambda/factor
        l_max += deltaLambda/factor
        if res == 0.0:
            res = deltaLambda/20000

        lambdas = np.arange(l_min, l_max, res)

        ''' Loops through each λ on range to find stable/unstable intervals '''
        is_insert_nsd, is_insert_ind, is_insert_psd = False, False, False
        nsd_intervals, ind_intervals, psd_intervals = [], [], []
        for l in lambdas:

            eigS = self.stability(l)

            ''' Negative definite (stability) intervals '''
            if np.all( eigS < 0.0 ):
                if not is_insert_nsd:
                    nsd_intervals.append([ l, np.inf ])
                is_insert_nsd = True
            elif is_insert_nsd:
                nsd_intervals[-1][1] = l
                is_insert_nsd = False

            ''' Indefinite (instability) intervals '''
            if np.any( eigS > 0.0 ) and np.any( eigS < 0.0 ):
                if not is_insert_ind:
                    ind_intervals.append([ l, np.inf ])
                is_insert_ind = True
            elif is_insert_ind:
                ind_intervals[-1][1] = l
                is_insert_ind = False

            ''' Positive definite (instability) intervals '''
            if np.all( eigS > 0.0 ):
                if not is_insert_psd:
                    psd_intervals.append([ l, np.inf ])
                is_insert_psd = True
            elif is_insert_psd:
                psd_intervals[-1][1] = l
                is_insert_psd = False

        strip_size = (q_max - q_min)/40
        strip_alpha = 0.6

        ''' Plot nsd strips (stable) '''
        for interval in nsd_intervals:
            xy = (interval[0], -strip_size/2)
            if interval[1] == np.inf: interval[1] = l_max
            length = interval[1] - interval[0]
            rect = patches.Rectangle(xy, length, strip_size, facecolor=mcolors.TABLEAU_COLORS['tab:orange'], alpha=strip_alpha)
            ax.add_patch(rect)

        ''' Plot indefinite strips (instable) '''
        for interval in ind_intervals:
            xy = (interval[0], -strip_size/2)
            if interval[1] == np.inf: interval[1] = l_max
            length = interval[1] - interval[0]
            rect = patches.Rectangle(xy, length, strip_size, facecolor=mcolors.TABLEAU_COLORS['tab:gray'], alpha=strip_alpha)
            ax.add_patch(rect)

        ''' Plot psd strips (instable) '''
        for interval in psd_intervals:
            xy = (interval[0], -strip_size/2)
            if interval[1] == np.inf: interval[1] = l_max
            length = interval[1] - interval[0]
            rect = patches.Rectangle(xy, length, strip_size, facecolor=mcolors.TABLEAU_COLORS['tab:cyan'], alpha=strip_alpha)
            ax.add_patch(rect)

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

        ''' Sets axes limits and legends '''
        ax.set_xlim(l_min, l_max)
        ax.set_ylim(q_min, q_max)
        ax.legend()

    def compatibilize(self, plant: DynamicSystem, clf_dict: dict, p = 1.0):
        '''
        Algorithm for finding a compatible CLF shape.
        '''
        if isinstance(plant, LinearSystem):
            A, B = plant._A, plant._B
            G = B @ B.T
        else:
            raise NotImplementedError("Currently, compatibilization is implemented linear systems only.")

        Hvfun = clf_dict["Hv_fun"]      # function for computing Hv
        x0 = clf_dict["center"]         # clf center

        def cost(var: np.ndarray):
            '''
            Objective function: find closest compatible CLF shape.
            '''
            eps = 1e-1
            Hv = Hvfun(var)
            cost_val = np.linalg.norm( Hv - clf_dict["Hv"] ) + eps*np.linalg.trace(Hv)

            return cost_val

        def compatibility_eigs(var: np.ndarray):
            '''
            Returns eigenvalues of
            '''
            Hv = Hvfun(var)
            eigHv = np.linalg.eigvals(Hv)
            # print(f"Eigs of Hv = {eigHv}")

            newN = p * G @ Hv - A
            self.update( N=newN )

            C = self.compatibility_matrix.sos_decomposition()

            eigC = np.linalg.eigvals(C)
            eigC.sort()
            # print(f"Eigens of C = {eigC}")

            return eigC

        var0 = sym2vector(clf_dict["Hv"])

        # n = self.dim
        # ndof = int(n*(n+1)/2)
        # var0 = np.random.randn(ndof)

        constr = [ {"type": "ineq", "fun": compatibility_eigs} ]
        sol = minimize( fun=cost, x0=var0, constraints=constr, options={"disp": True, "maxiter": 1000} )

        results = {
                    "Hv": Hvfun(sol.x),
                    "cost": cost(sol.x),
                    "compatibility": compatibility_eigs(sol.x)
                   }

        return results

    def old_compatibilize(self, plant: DynamicSystem, clf_dict: dict, cbf_dict, p = 1.0):
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

                real_zeros = [ z.real for z in zeros if np.abs(z.imag) < self.real_tol ]
                real_zeros = [ z.real for z in zeros if np.abs(z.imag) < self.real_tol ]
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