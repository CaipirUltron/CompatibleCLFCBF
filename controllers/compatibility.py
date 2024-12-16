import math
import numpy as np
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from matplotlib.axes import Axes
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial.polynomial import polydiv

from copy import copy
from scipy import signal
from scipy.optimize import fsolve, minimize
from scipy.linalg import null_space, inv

from common import *
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

        self.perturbing = False

        ''' Class parameters '''
        self.trim_tol = 1e-8
        self.real_tol = 1e-6
        self.comp_poly_leading_coef = 0.01
        self.compatibility_eps = 1.0       # should be > 1

        ''' Stability matrix S(λ) '''
        self.stb_dim = self.dim - 1
        self.Smatrix: MatrixPolynomial = MatrixPolynomial.zeros(size=(self.stb_dim, self.stb_dim))

        # self.N = self.stb_dim - 1
        # self.num_fixed_pts = 0

        self._compute_polynomials()

        if self.is_compatible():
            print("Q-function is compatible.")
        else:
            print("Q-function is not compatible.")

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

        ''' Computation of the zero-polynomial, for computing the boundary equilibrium points '''
        self.zero_poly = self.n_poly - self.d_poly
        self.diff_zero_poly = self.zero_poly.deriv()

        ''' Computation of stability properties of the boundary equilibrium points '''
        self.divisor_poly, self.v_poly = self._v_poly()

        if not self.perturbing:
            self._stability_matrix()

    def _init_convex_opt(self, num_factors, fixed_pts):
        ''' 
        Setup CVXPY parameters and variables.
        '''
        self.sos_dim = num_factors + 1
        self.sos_shape = (self.sos_dim, self.sos_dim)

        self.sos = cvx.Variable( shape=self.sos_shape, PSD=True )

        D = np.diag([k for k in range(1,self.sos_dim)])
        D = np.hstack([D, np.zeros((self.sos_dim-1,1))])
        D = np.vstack([np.zeros((1,self.sos_dim)),D])

        def kern(l:float) -> np.ndarray:
            return np.array([ l**k for k in range(self.sos_dim) ])
        
        # Polynomial must be monic
        CVX_constraints = [ self.sos[-1,-1] == 1 ]

        # This adds the fixed point constraints
        for pt in fixed_pts:
            Lambda = kern(pt)
            CVX_constraints += [ Lambda.T @ ( D.T @ self.sos + self.sos @ D ) @ Lambda == 0 ]

        self.CVX_cost = cvx.norm( self.sos )
        self.CVX_problem = cvx.Problem( cvx.Minimize( self.CVX_cost ), constraints=CVX_constraints )

    def _v_poly(self) -> tuple[Poly, np.ndarray[Poly]]:
        '''
        Compute the v(λ) polynomial
        '''
        return self._v_poly_derivative(order=0)

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

    def _intervals(self):
        ''' 
        Compute stability intervals using the gen. eigenvalues of S(λ) (|S(λ)| = 0)
        '''
        def interval_type(limits: tuple):
            '''
            Returns the amount of positive/negative eigenvalues of S at a given interval.
            '''
            if len(limits) != 2 or limits[0] >= limits[1]:
                raise Exception("Invalid interval limits.")

            displ_dist = 10.0
            if limits[0] == -np.inf and limits[0] == +np.inf:
                l = np.random.randn()
            elif limits[0] == -np.inf:
                l = limits[1] - displ_dist
            elif limits[1] == np.inf:
                l = limits[0] + displ_dist
            else:
                l = np.mean(limits)

            eigS_at_test_pt = np.linalg.eigvals( self.Smatrix(l) )
            pos_num = int(np.sum([ 1 for eig in eigS_at_test_pt if eig > 0]))
            neg_num = self.stb_dim - pos_num
            return (pos_num, neg_num)

        real_eigS = self.Smatrix_pencil.real_eigen()
        divisions = [-np.inf] + [ float(eig.eigenvalue) for eig in real_eigS if eig.eigenvalue != np.inf ] + [np.inf]

        self.intervals = []
        self.stability_intervals = []
        for k in range(len(divisions)-1):
            div1 = divisions[k]
            div2 = divisions[k+1]
            limits = (div1, div2)
            interval = { "limits": limits, 
                         "type": interval_type(limits),
                         "length": limits[1]-limits[0]
                         }
            self.intervals.append( interval )

            if div2 < 0: continue
            if div1 < 0: div1 = 0.0

            limits = (div1, div2)
            if interval_type(limits) != (0, self.stb_dim):
                stability = 'unstable'
            else:
                stability = 'stable'

            interval = { "limits": limits, 
                         "stability": stability,
                         "length": abs(limits[1]-limits[0])
                         }
            self.stability_intervals.append( interval )

        # for k, interval in enumerate(self.intervals):
        #     typ = interval["type"]
        #     limits = interval["limits"]
        #     print(f"{k+1} interval = {limits} is {typ}")

        # for k, interval in enumerate(self.stability_intervals):
        #     stab = interval["stability"]
        #     limits = interval["limits"]
        #     print(f"{k+1} interval = {limits} is {stab}")

    def _stability_matrix(self) -> MatrixPolynomial:
        ''' 
        Computes the stability polynomial matrix.
        '''

        ''' Computes R(λ), the nullspace matrix polynomial to ∇h(λ) '''
        nablah = ( self.H @ self.v_poly )
        gradNull = nullspace( nablah )
        coefs, symbol = to_coef(gradNull)
        nullspace_poly_deg = len(coefs)-1

        # If the nullspace degree is zero (degenerate case), PERTURB the parameters of the Q-function to generate a new nullspace
        if nullspace_poly_deg == 0:
            
            self.perturbing = True

            perturb = 1e-10
            self.auxM, self.auxN, self.auxH, self.auxw = self.pencil.M, self.pencil.N, self.H, self.w

            newM = self.pencil.M + perturb*np.random.randn(*self.pencil.shape)
            newN = self.pencil.N + perturb*np.random.randn(*self.pencil.shape)

            newH = self.H + perturb*np.random.randn(self.dim)
            neww = self.w + perturb*np.random.randn(self.dim)

            self.update(M=newM, N=newN, H=newH, w=neww)
            nablah = ( self.H @ self.v_poly )
            gradNull = nullspace( nablah )

            self.update(M=self.auxM, N=self.auxN, H=self.auxH, w=self.auxw)
            self.perturbing = False

        self.Rmatrix = gradNull

        ''' Computes the stability matrix polynomial from N(λ) '''
        self.Psym = self.pencil.symmetric()
        self.Smatrix.update( self.Rmatrix.T @ self.Psym @ self.Rmatrix )

        if hasattr(self, "Smatrix_pencil"):
            newM, newN = companion_form( self.Smatrix.poly_array )
            self.Smatrix_pencil.update( M = newM, N = newN )
        else:
            self.Smatrix_pencil = self.Smatrix.companion_form()

        self._intervals()
        self.compatibility_barrier(verbose=True)

    def compatibility_barrier(self, verbose=False):
        ''' 
        Barrier function for compatibility 
        '''
        N = self.dim
        self.compatibility_poly = Poly([self.comp_poly_leading_coef])
        intervals = copy(self.intervals)            

        ''' First, deal with the stable intervals '''
        to_be_rem = []
        for interval in intervals:

            if interval["type"] != (0, self.stb_dim):           # ignores non-nsd intervals
                continue

            roots = [ max(l, 0.0) for l in interval["limits"] ]
            self.compatibility_poly *= Poly.fromroots(roots)

            N -= 1
            to_be_rem.append(interval)

        for item in to_be_rem:
            intervals.remove(item)

        ''' Next, deal with complex conjugate pairs '''
        def are_conj(num1, num2):
            return num1.imag == -num2.imag and num1.real == num2.real

        eigS = self.Smatrix_pencil.eigens
        complex_conjugates = []
        for eig in eigS:
            if np.iscomplex(eig.eigenvalue) and not np.any([ are_conj(eig.eigenvalue, num) for num in complex_conjugates ]):
                complex_conjugates.append(eig.eigenvalue)

                a = eig.eigenvalue.real
                b = eig.eigenvalue.imag

                h0 = a**2 + b**2
                h1 = -2*a
                self.compatibility_poly *= Poly([h0, h1, 1.0])

        ''' Deal with finite psd intervals '''
        to_be_rem = []
        for interval in intervals:

            if interval["type"] != (self.stb_dim, 0) or interval["length"] == np.inf:           # ignores non-psd intervals
                continue

            a = interval["limits"][0]
            b = interval["limits"][1]
            l = interval["length"]
            
            h0 = l + ((a+b)/2)**2
            h1 = -(a+b)
            self.compatibility_poly *= Poly([h0, h1, 1.0])

            to_be_rem.append(interval)

        for item in to_be_rem:
            intervals.remove(item)

        ''' Finally, deal with remaining connected intervals '''
        visited = [ False for _ in range(len(intervals)) ]
        for k in range(len(intervals)-1):
            
            int1 = intervals[k]
            int2 = intervals[k+1]
            if not np.all(visited[k:k+2]) and int1["type"] == int2["type"]:

                a = int1[0]
                b = int2[1]
                l = b-a
                
                h0 = l + (0.5*(a+b))**2
                h1 = -(a+b)
                self.compatibility_poly *= Poly([h0, h1, 1.0])

                # fixed_pts.append( np.mean([ int1[0], int2[1] ]) )
                visited[k] = True
                visited[k+1] = True

        print(f"Compatibility polynomial: \n{self.compatibility_poly}")
        print(f" with roots = {self.compatibility_poly.roots()}")

    def _comp_nonconvex_opt(self, verbose=False):
        ''' 
        Nonconvex compatible optimization.
        Variables:
        - delta: float
        - Lambda: ( seff.null_dim x self.Cdim ) np.ndarray
        - C = C' (compatibility SOS matrix)
        '''
        def getVariables(var: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
            ''' var = [ delta, Lambda_flatten, symmC_flatten ] '''
            delta = var[0]

            Lambda_dim = self.null_dim * self.Cdim
            Lambda_flatten = var[1:Lambda_dim+1]
            Lambda = Lambda_flatten.reshape(self.null_dim, self.Cdim)

            symm_sosC_dim = int(self.sosC_dim*(self.sosC_dim+1)/2)
            symm_sosC_flatten = var[Lambda_dim+1:symm_sosC_dim+Lambda_dim+1]
            sosC = vector2sym(symm_sosC_flatten)

            return delta, Lambda, sosC
        
        def toVar(delta: float, Lambda: np.ndarray, sosC: np.ndarray) -> np.ndarray:
            Lambda_flatten = Lambda.flatten()
            C_flatten = sym2vector(sosC)
            var = np.hstack([ delta, Lambda_flatten, C_flatten ])
            return var

        def extract_blocks(a: np.ndarray, blocksize: tuple, keep_as_view=True) -> np.ndarray:
            M,N = a.shape
            b0, b1 = blocksize
            if keep_as_view==0:
                return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2).reshape(-1,b0,b1)
            else:
                return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2)

        def cost(var: np.ndarray) -> float:
            '''
            Minimizes square of slack variable (feasibility problem)
            '''
            delta, Lambda, sosC = getVariables(var)
            return delta**2 + np.linalg.norm( Lambda.T @ Lambda )

        def sos_constraint(var: np.ndarray) -> np.ndarray:
            '''
            SOS constraints on problem variables
            '''
            delta, Lambda, sosC = getVariables(var)
            sosCblks = extract_blocks(sosC, self.Cshape)

            ''' Each loop constructs one SOS constraint '''
            sos_errors = []
            for locs, d, aug_lScoef in zip( self.sos_locs, self.zero_poly.coef, self.aug_lSmatrix.coef ):
                CERROR: np.ndarray = sum([ sosCblks[index] if index[0]==index[1] else sosCblks[index] + sosCblks[index].T for index in locs ]) - ( d * self.E + Lambda.T @ aug_lScoef @ Lambda )
                sos_errors += CERROR.flatten().tolist()

            print(f"{sos_errors}")

            return np.array(sos_errors)

        def PSD_constraint(var: np.ndarray) -> np.ndarray:
            '''
            PSD constraint on compatibility matrix
            '''
            delta, Lambda, sosC = getVariables(var)

            PSD = sosC + delta * np.eye(self.sosC_dim)
            eigPSD = np.linalg.eigvals(PSD)
            eigPSD.sort()

            return eigPSD

        constraints = [ {"type": "eq", "fun": sos_constraint} ]
        constraints += [ {"type": "ineq", "fun": PSD_constraint} ]
        sol = minimize( fun=cost, x0=toVar(self.delta, self.Lambda, self.sosC), constraints=constraints, method='SLSQP', options={"disp": verbose, "maxiter": 1000} )
        self.delta, self.Lambda, self.sosC = getVariables(sol.x)

        if verbose:
            print(f"Nonvex results: δ = {self.delta}, Λ = \n{self.Lambda}")

    def is_compatible(self):
        ''' 
        Test if Q-function is compatible or not.
        Q-function is compatible iff there exist NO roots of n(λ) - |P(λ)|² inside the stable intervals.
        '''
        # comp = self.compatibility()
        # print(f"Compatibility function returned: {comp}")

        for interval in self.stability_intervals:
            if interval["stability"] == 'stable':
                for root in self.zero_poly.roots():
                    if np.abs(root.imag) > 1e-6:
                        continue
                    limits = interval["limits"]
                    if root >= limits[0] and root <= limits[1]:
                        return False

        return True

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
        the Q-function by solving q(λ) = n(λ)/|P(λ)|² = 1, that is, by finding real roots of n(λ) - |P(λ)|².

        Returns: a list of dictionaries with all boundary equilibrium solutios and their stability numbers.
        '''

        # Gets real roots of n(λ) - |P(λ)|² and finds possibly duplicated roots ( canceling poles and zeros of q(λ) ) 
        zeros = self.zero_poly.roots()
        real_zeros = np.array([ z.real for z in zeros if np.abs(z.imag) < self.real_tol and z.real >= 0.0 ])
        real_zeros.sort()
        duplicates = [ np.any( np.abs(np.delete(real_zeros,k) - z) <= 1e-4 ) for k, z in enumerate(real_zeros) ]

        # Adds equilibrium solutions, distinguishing degenerate ones ( when poles and zeros of q(λ) cancel each other )
        equilibrium_sols = []
        for is_repeated, z in zip(duplicates, real_zeros):
            if np.any([ sol["lambda"] == z for sol in equilibrium_sols ]):
                continue
            if is_repeated:
                q_n, r_n = polydiv( self.n_poly.coef, Poly.fromroots(z).coef )
                q_d, r_d = polydiv( self.d_poly.coef, Poly.fromroots(z).coef )
                q_val = Poly(q_n)(z)/Poly(q_d)(z)
                if q_val < 1.0:
                    equilibrium_sols.append( {"lambda": z, "degenerate": True} )
                continue
            equilibrium_sols.append( {"lambda": z, "degenerate": False} )

        # Computes stability from the S(λ) matrix ( TO DO: fix stability computation in degenerated cases )
        for sol in equilibrium_sols:
            eigS = self.stability(sol["lambda"])
            sol["stability"] = max(eigS)

        return equilibrium_sols

    def stability(self, l):
        '''
        Returns the eigenvalues of the stability matrix S(λ) computed at λ
        '''
        stabilityEigs = np.array([ eig for eig in np.linalg.eigvals( self.Smatrix(l) ) if np.abs(eig.imag) < self.real_tol ])
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

    def plot(self, ax: Axes, res: float = 0.0, q_limits=(-5, 5)):
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

        for eig in self.Smatrix_pencil.real_eigen():
            if np.abs(eig.eigenvalue) < 1e+12: 
                lambdaRange.append(eig.eigenvalue)

        ''' Using range min, max values, generate λ range to be plotted '''
        factor = 10
        l_min, l_max = -100,  100
        l_min, l_max = min(lambdaRange), max(lambdaRange)

        deltaLambda = l_max - l_min
        l_min -= deltaLambda/factor
        l_max += deltaLambda/factor
        if res == 0.0:
            res = deltaLambda/1e4

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

        strip_size = (q_max - q_min)/50
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
        # ax.plot( [ 0.0, 0.0 ], [ q_min, q_max ], 'k' )          # vertical axis

        ''' Plot the solution line q(λ) = 1 '''
        ax.plot( lambdas, [ 1.0 for l in lambdas ], 'r--' )

        ''' Plot the Q-function q(λ) and zero polynomial n(λ) - |P(λ)|² '''
        # q_array = [ self.n_poly(l) / self.d_poly(l) for l in lambdas ]
        # ax.plot( lambdas, q_array, label='q(λ)' )

        z_array = [ self.zero_poly(l) for l in lambdas ]
        ax.plot( lambdas, z_array, color='g', label='n(λ) - |P(λ)|²' )

        ''' Plots equilibrium (stable/unstable) λ solutions satisfying q(λ) = 1 '''
        for k, sol in enumerate(sols):
            stability = sol["stability"]
            label_txt = f"{k+1}"
            ax.text(sol["lambda"], 1.0, f"{k+1}", color='k', fontsize=12)
            if stability > 0:
                label_txt += f" unstable ({stability:1.5f})"
                ax.plot( sol["lambda"], 1.0, 'bo', label=label_txt) # unstable solutions
            if stability < 0:
                label_txt += f" stable ({stability:1.5f})"
                ax.plot( sol["lambda"], 1.0, 'ro', label=label_txt ) # stable solutions

        cp_array = [ self.compatibility_poly(l) for l in lambdas ]
        ax.plot( lambdas, cp_array, color='r', label='h(λ)' )

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
            cost_val = np.linalg.norm( Hv - clf_dict["Hv"] )

            return cost_val

        def compatibility_eigs(var: np.ndarray):
            '''
            Returns eigenvalues of
            '''
            Hv = Hvfun(var)
            eigHv = np.linalg.eigvals(Hv)
            print(f"Eigs of Hv = {eigHv}")

            newN = p * G @ Hv - A
            self.update( N=newN )

            C = self.augCmatrix.sos_decomposition()

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