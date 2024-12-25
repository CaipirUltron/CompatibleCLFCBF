import math
import numpy as np

import matplotlib.patches as patches
import matplotlib.colors as mcolors

from matplotlib.axes import Axes
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial.polynomial import polydiv, polyder, polyint

from typing import Callable
from copy import copy
from scipy import signal
from scipy.optimize import fsolve, minimize
from scipy.linalg import null_space, inv

from common import *
from dynamic_systems import LinearSystem, DriftLess
from functions import QuadraticLyapunov, QuadraticBarrier

def inside(num, interval):
    t1 = num >= interval[0] and num <= interval[1]
    t2 = num >= interval[1] and num <= interval[0]
    return t1 or t2

def continuous_composition(k, *args, **kwargs):
    ''' 
    Continuously compose any number of functions (with weights).
    '''
    weigths = [ 1.0 for _ in args ]
    if "weights" in kwargs.keys():
        weigths = kwargs["weights"]
        if len(weigths) != len(args):
            raise TypeError("Number of weigths must be the same as passed arguments.")

    if len(args) > 0:
        exps = [ w * np.exp( -k*arg ) for arg, w in zip(args, weigths) ]
        return -(1/k)*np.log(sum(exps))
    else:
        return 1.0

class QFunction():
    '''
    General class for Qfunctions q(λ) = v(λ)' H v(λ), where P(λ) v(λ) = w
    and P(λ) is a regular linear matrix pencil.
    '''
    def __init__(self, plant: LinearSystem | DriftLess, clf: QuadraticLyapunov, cbf: QuadraticBarrier, p=1):

        if not isinstance(plant, (LinearSystem, DriftLess) ):
            raise NotImplementedError("Q-function implemented for LTI and driftless full-rank systems only.")
        
        if not isinstance(clf, QuadraticLyapunov):
            raise Exception("Q-function implemented for quadratic CLFs only.")

        if not isinstance(cbf, QuadraticBarrier):
            raise Exception("Q-function implemented for quadratic CBFs only.")

        self.plant = plant
        self.clf = clf
        self.cbf = cbf
        self.p = p

        self.Hv = self.clf.H
        self.H = self.cbf.H

        if isinstance(self.plant, LinearSystem):
            A, B = self.plant._A, self.plant._B
            G = B @ B.T
            self.M = G @ self.H
            self.N = self.p * G @ self.Hv - A
        if isinstance(self.plant, DriftLess):
            self.M = self.H
            self.N = self.p @ self.Hv

        self.w = self.N @ ( cbf.center - self.clf.center )
        self.pencil = MatrixPencil(self.M, self.N)
        self._verify(self.pencil, self.H, self.w)

        self.perturbing = False

        ''' Class parameters '''
        self.real_tol = 1e-6
        self.epsilon = 1e-3
        self.composite_kappa = 20
        self.normalization_kappa = 1e-4
        self.comptol = 1.0                  # Should be a small positive value

        ''' Stability matrix S(λ) '''
        self.stb_dim = self.dim-1
        self.Smatrix: MatrixPolynomial = MatrixPolynomial.zeros(size=(self.stb_dim, self.stb_dim))

        self._compute_polynomials()

        if self.satisfy_clf():
            print("Passed V satisfies the CLF conditon.")
        else:
            print("Passed V does not satisfies the CLF conditon.")

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
        self.determinant, self.adjoint = self.pencil.inverse()
        self.n_poly = ( self.w.T @ self.adjoint.T @ self.H @ self.adjoint @ self.w )
        self.d_poly = ( self.determinant**2 )

        # Normalize numerator/denominator coefficients with the mean value of all coefficients
        norm_coef = np.mean([ np.abs(c) for c in self.n_poly.coef ]+[ np.abs(c) for c in self.d_poly.coef ])
        self.n_poly = self.n_poly/norm_coef
        self.d_poly = self.d_poly/norm_coef

        ''' Computation of the zero-polynomial, for computing the boundary equilibrium points '''
        self.zero_poly = self.n_poly - self.d_poly
        self.diff_zero_poly = self.zero_poly.deriv()

        if self.zero_poly(0) >= 0: 
            tol = 1 + self.comptol
        else: 
            tol = 1 - self.comptol

        self.safe_zero_poly = self.n_poly - tol * self.d_poly
        self.diff_safe_zero_poly = self.safe_zero_poly.deriv()

        ''' Computation of stability properties of the boundary equilibrium points '''
        self.divisor_poly, self.v_poly = self._v_poly()

        if not self.perturbing:
            self._stability_matrix()

        ''' Computation of equilibrium points and their stability '''
        self._equilibria()

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
        # self.det, self.adjoint = self.pencil.inverse()
        v = math.factorial(order) * self.adjoint @ self.w

        power = np.eye(self.pencil.shape[0])
        for k in range(order):
            power @= - self.adjoint @ self.pencil.M

        div = self.determinant**(order+1)
        v = power @ v
        return div, v

    def _definitess_intervals(self):
        ''' 
        Computes stability intervals using the gen. eigenvalues of S(λ) (|S(λ)| = 0)
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
        divisions = [-np.inf] + [ float(eig.eigenvalue) for eig in real_eigS if np.abs(eig.eigenvalue) != np.inf ] + [np.inf]
        
        self.intervals = []
        self.stability_intervals = []
        for k in range(len(divisions)-1):
            div1 = divisions[k]
            div2 = divisions[k+1]
            limits = (div1, div2)
            interval = { "limits": limits, 
                         "type": interval_type(limits),
                         "length": limits[1]-limits[0] }
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

    def _stability_conj_pairs(self):
        '''
        Computes the complex conjugate pairs among the stability eigenvalues
        '''
        def are_conj(num1, num2):
            return num1.imag == -num2.imag and num1.real == num2.real

        self.stability_conj_pairs = []
        for eig in self.Smatrix_pencil.eigens:
            if np.iscomplex(eig.eigenvalue) and not np.any([ are_conj(eig.eigenvalue, num) for num in self.stability_conj_pairs ]):
                self.stability_conj_pairs.append(eig.eigenvalue)

    def _stability_matrix(self) -> MatrixPolynomial:
        ''' 
        Computes the stability polynomial matrix.
        '''

        # Computes R(λ), the nullspace matrix polynomial to ∇h(λ)
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

        # Computes the stability matrix polynomial from N(λ)
        self.Psym = self.pencil.symmetric()
        self.Smatrix.update( self.Rmatrix.T @ self.Psym @ self.Rmatrix )

        if hasattr(self, "Smatrix_pencil"):
            newM, newN = companion_form( self.Smatrix.poly_array )
            self.Smatrix_pencil.update( M = newM, N = newN )
        else:
            self.Smatrix_pencil = self.Smatrix.companion_form()

        self._definitess_intervals()
        self._stability_conj_pairs()

    def _equilibria(self) -> list[dict]:
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
        self.equilibrium_sols = []
        for is_repeated, z in zip(duplicates, real_zeros):
            if np.any([ sol["lambda"] == z for sol in self.equilibrium_sols ]):
                continue
            if is_repeated:
                q_n, r_n = polydiv( self.n_poly.coef, Poly.fromroots(z).coef )
                q_d, r_d = polydiv( self.d_poly.coef, Poly.fromroots(z).coef )
                q_val = Poly(q_n)(z)/Poly(q_d)(z)
                if q_val < 1.0:
                    self.equilibrium_sols.append( {"lambda": float(z), "degenerate": True} )
                continue
            self.equilibrium_sols.append( {"lambda": float(z), "degenerate": False} )

            l = self.equilibrium_sols[-1]["lambda"]
            v_array = np.array([ poly(l) for poly in self.v_poly ])
            self.equilibrium_sols[-1]["point"] = ( v_array/self.divisor_poly(l) + self.cbf.center ).tolist()

        # Computes stability from the S(λ) matrix ( TO DO: fix stability computation in degenerated cases )
        for sol in self.equilibrium_sols:
            l = sol["lambda"]
            eigS = self.stability(l)
            sol["stability"] = float(max(eigS))

    def _qvalue(self, l: float, H: list | np.ndarray, w: list | np.ndarray) -> float:
        '''
        Computes the q-function value q(λ) = n(λ)/d(λ) for a given λ (for DEBUG only)
        '''
        P = self.pencil(l)
        v = inv(P) @ self.w
        return v.T @ self.H @ v

    def _interval_barrier(self, interval, typ='above'):
        ''' Defines barrier function for interval '''

        val1 = float(self.safe_zero_poly(interval[0]))
        val2 = float(self.safe_zero_poly(interval[1]))

        extremal_candidates = [ val1, val2 ]
        for root in self.diff_safe_zero_poly.roots():
            if np.isreal(root) and inside(root.real, interval):
                val = float(self.safe_zero_poly(root.real))
                extremal_candidates.append( val )

        maximum, minimum = max(extremal_candidates), min(extremal_candidates)
        length = np.abs(interval[1] - interval[0])
        k = self.normalization_kappa
        if typ == 'above':
            return math.tanh( + k * length * minimum ), length
        if typ == 'below':
            return math.tanh( - k * length * maximum ), length

    def composite_barrier(self):

        barrier_limits = []
        for interval in self.intervals:

            if interval["type"] != (0, self.stb_dim):           # ignores non-nsd intervals
                continue

            limits = interval["limits"]
            if limits[1] < 0:
                min_bound, max_bound = 0.0, 0.0
            elif limits[0] == -np.inf: 
                min_bound, max_bound = 0.0, limits[1]
            else:
                min_bound, max_bound =  limits[0], limits[1]
            barrier_limits.append( [min_bound, max_bound] )      

        if self.zero_poly(0) >= 0: typ = 'above'
        else: typ = 'below'

        barrier_vals, weights = [], []
        for barrier_lim in barrier_limits:
            h, length = self._interval_barrier(barrier_lim, typ)
            barrier_vals.append(h)
            weights.append(length)

        # return continuous_composition(self.composite_kappa, *barrier_vals, weights=weights)
        return continuous_composition(self.composite_kappa, *barrier_vals)

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

    def satisfy_clf(self):
        ''' Checks if given CLF satisfy the CLF condition '''

        if isinstance(self.plant, LinearSystem):
            A, B = self.plant._A, self.plant._B
        else:
            raise NotImplementedError("CLF condition test not implemented for non-LTI systems.")
        
        Hv = self.clf.H
        Q = Hv @ A + A.T @ Hv
        eigsQ = np.linalg.eigvals(Q)

        return np.all(eigsQ < 0)

    def is_compatible(self):
        ''' 
        Test if Q-function is compatible or not.
        Q-function is compatible iff there exist NO roots of n(λ) - |P(λ)|² inside the stable intervals.
        '''
        for interval in self.stability_intervals:
            if interval["stability"] == 'stable':
                for root in self.zero_poly.roots():
                    if np.abs(root.imag) > 1e-6:
                        continue
                    limits = interval["limits"]
                    if root >= limits[0] and root <= limits[1]:
                        return False

        return True

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

    def compatibilize(self, verbose=False):
        '''
        Computes a compatible CLF.
        '''
        Hv0 = self.Hv

        if isinstance(self.plant, LinearSystem):
            A, B = self.plant._A, self.plant._B
        if isinstance(self.plant, DriftLess):
            A, B = np.zeros((self.plant.n)), self.eye(self.plant.n)

        G = B @ B.T

        def ellipsoid_vol(Hv):
            ''' Proportional to the ellipsoid volume '''
            return - np.log( np.linalg.det(Hv) )

        def cost(var: np.ndarray):
            ''' Seeks to minimize the distance from reference CLF and ellipsoid volume '''
            Hv = self.clf.param2Hv(var)
            # print( f"Eigenvalues of Hv = { np.linalg.eigvals(Hv) }" )

            cost = np.linalg.norm( Hv - Hv0, 'fro' )
            cost += ellipsoid_vol(Hv)
            return cost

        def compatibility(var: np.ndarray):
            ''' Returns compatibility barrier '''

            Hv = self.clf.param2Hv(var)
            new_N = self.p * G @ Hv - A
            new_w = new_N @ ( self.cbf.center - self.clf.center )
            self.update( N=new_N, w=new_w )

            constr = self.composite_barrier()
            return constr

        def clf_condition(var: np.ndarray):
            ''' Satisfies the CLF condition '''

            Hv = self.clf.param2Hv(var)
            eigs = np.linalg.eigvals( Hv @ A + A.T @ Hv )
            return - eigs

        var0 = self.clf.param
        constr = []
        constr += [ {"type": "ineq", "fun": compatibility} ]
        constr += [ {"type": "ineq", "fun": clf_condition} ]
        sol = minimize( fun=cost, x0=var0, constraints=constr, method='SLSQP', options={"disp": True, "maxiter": 1000} )

        # DO NOT update the self.clf after compatibilization;
        # Instead, update only the Q-function parameters
        Hv = self.clf.param2Hv(sol.x)
        new_N = self.p * G @ Hv - A
        new_w = new_N @ ( self.cbf.center - self.clf.center )
        self.update( N=new_N, w=new_w )

        results = {"Hv": Hv,
                   "cost": cost(sol.x),
                   "compatibility": compatibility(sol.x),
                   "clf_conditon": clf_condition(sol.x)}
        
        if verbose:
            print("Compatibilization results: ")
            print(results)

        return results

    def init_graphics(self, ax: Axes, res=0.01):
        ''' Initialize graphical objects '''

        # Get horizontal and vertical ranges for plotting

        eq_sols = [ root.real for root in self.zero_poly.roots() if np.isreal(root) ]
        realeigs = [ eig.eigenvalue for eig in self.Smatrix_pencil.real_eigen() if np.abs(eig.eigenvalue) != np.inf ]
        complexeigs = [ eig.real for eig in self.stability_conj_pairs ]
        lambda_range = [0.0] + eq_sols + realeigs + complexeigs

        lmin, lmax = max(min(lambda_range), -100), min(max(lambda_range), +100)

        z_range = [ self.zero_poly(root.real) for root in self.diff_zero_poly.roots() if np.isreal(root) ]
        z_range += [ self.zero_poly(lmin) ]
        z_range += [ self.zero_poly(lmax) ]
        zmin, zmax = max(min(z_range),-100), min(max(z_range),+100)

        self.plot_configs = {"hor": [lmin-5, lmax+5],
                             "ver": [zmin-10, zmax+10],
                             "inertia": False }

        self.strip_colors = {"nsd": np.array(mcolors.to_rgb(mcolors.TABLEAU_COLORS['tab:red'])),
                             "psd": np.array(mcolors.to_rgb(mcolors.TABLEAU_COLORS['tab:cyan'])),
                             "indef": np.array(mcolors.to_rgb(mcolors.TABLEAU_COLORS['tab:gray']))}

        self.lambda_res = res
        zmin, zmax = self.plot_configs["ver"]
        self.strip_size = np.abs(zmax-zmin)/20

        hor_limits = self.plot_configs["hor"]
        self.lambda_array = np.arange(hor_limits[0], hor_limits[1], self.lambda_res)
        self.q_plot, = ax.plot([],[],'b',lw=0.8, label='$q(\lambda) = \dfrac{n(\lambda)}{|P(\lambda)|^2}$')
        self.z_plot, = ax.plot([],[],'g',lw=0.8, label='$z(\lambda) = n(\lambda) - |P(\lambda)|^2$')
        self.ones_plot, = ax.plot(self.lambda_array,[ 1.0 for l in self.lambda_array ],'k--',lw=1.0)

        self.stable_pts, = ax.plot([], [], 'or' )
        self.unstable_pts, = ax.plot([], [], 'ob' )
        self.equilibrium_texts = []
        for _ in range(self.zero_poly.degree()):
            self.equilibrium_texts.append( ax.text(0.0, 0.0, str(""), fontsize=10) )

        self.default_rect = patches.Rectangle((0.0, -self.strip_size/2), 0, self.strip_size, facecolor=self.strip_colors["indef"], alpha=.6)
        self.strip_rects, self.interval_texts = [], []
        for _ in range(len(self.Smatrix_pencil.eigens)+1):
            handle = ax.add_patch( copy(self.default_rect) )
            self.strip_rects.append(handle)
            self.interval_texts.append( ax.text(0.0, 0.0, str(""), fontsize=10) )

        self.real_stability_pts, = ax.plot([], [], '*k', linewidth=1.0 )
        self.complex_stability_pts, = ax.plot([], [], '|k', linewidth=1.0 )

        ax.grid()
        ax.set_xlim(*self.plot_configs["hor"])
        ax.set_ylim(*self.plot_configs["ver"])
        ax.set_xlabel('$\lambda$', fontsize = 'small')
        ax.set_ylabel('', fontsize = 'small')
        ax.legend(loc='upper right', fontsize = 'small')

    def plot(self):
        '''
        Plots the Q-function for analysis.
        '''
        hor_limits = self.plot_configs["hor"]
        
        # Zero polynomial
        z_array = [ self.zero_poly(l) for l in self.lambda_array ]
        self.z_plot.set_data(self.lambda_array, z_array)

        # Q-function polynomial
        q_array = [ self.n_poly(l)/self.d_poly(l) for l in self.lambda_array ]
        self.q_plot.set_data(self.lambda_array, q_array)

        # Definiteness intervals
        num_updated = 0
        for k, interval in enumerate(self.intervals):

            limits = interval["limits"]
            # if limits[1] < hor_limits[0] or hor_limits[1] < limits[0]:
            #     continue

            typ = interval["type"]
            lmin = max(limits[0], hor_limits[0])
            lmax = min(limits[1], hor_limits[1])
            length = lmax - lmin

            self.strip_rects[k].set_x(lmin)
            self.strip_rects[k].set_width(length)
            num_updated += 1

            strip_color = (1/self.stb_dim) * ( self.strip_colors["psd"]*typ[0] + self.strip_colors["nsd"]*typ[1] )
            self.strip_rects[k].set_facecolor(strip_color)

            if self.plot_configs["inertia"]:
                interval_text = self.interval_texts[k]
                interval_text.set_text(f"{typ}")
                interval_text.set_x(lmin+0.5*length)
                interval_text.set_y( ((-1)**k)*(self.strip_size/2)*1.2 )

        for i in range(num_updated, len(self.strip_rects)):
            self.strip_rects[i].set_width(0)
            self.interval_texts[i].set_text("")

        # Real stability eigenvalues
        eig_array = [ eig.eigenvalue for eig in self.Smatrix_pencil.real_eigen() ]
        self.real_stability_pts.set_data(eig_array, np.zeros(len(eig_array)))

        # Complex conjugate stability eigenvalues
        if self.stability_conj_pairs:
            hor_array = np.hstack([ [eig.real, eig.real] for eig in self.stability_conj_pairs ])
            ver_array = np.hstack([ [eig.imag, -eig.imag] for eig in self.stability_conj_pairs ])
            self.complex_stability_pts.set_data(hor_array, ver_array)
        else:
            self.complex_stability_pts.set_data([], [])

        # Equilibrium points
        stable_eq_array = []
        unstable_eq_array = []
        for k, sol in enumerate( self.equilibrium_sols ):
            l = sol["lambda"]
            if sol["stability"] < 0:
                stable_eq_array.append(l)
            else:
                unstable_eq_array.append(l)
            self.equilibrium_texts[k].set_x(l)
            self.equilibrium_texts[k].set_text(f"{k+1}")

        self.stable_pts.set_data( stable_eq_array, np.zeros(len(stable_eq_array)) )
        self.unstable_pts.set_data( unstable_eq_array, np.zeros(len(unstable_eq_array)) )

        for i in range(k+1, len(self.equilibrium_texts)):
            self.equilibrium_texts[k].set_text(f"")

        # Returns graphical handlers
        graphical_elements = []
        graphical_elements.append( self.ones_plot )
        graphical_elements.append( self.z_plot )
        graphical_elements.append( self.q_plot )
        graphical_elements += self.strip_rects
        graphical_elements += self.interval_texts
        graphical_elements.append( self.real_stability_pts )
        graphical_elements.append( self.complex_stability_pts )
        graphical_elements.append( self.stable_pts )
        graphical_elements.append( self.unstable_pts )
        graphical_elements += self.equilibrium_texts

        return graphical_elements

# class CLFCBFPair():
#     '''
#     Class for a CLF-CBF pair. Computes the q-function, equilibrium points and critical points of the q-function.
#     '''
#     def __init__(self, clf, cbf):

#         self.eigen_threshold = 0.000001
#         self.update(clf = clf, cbf = cbf)

#     def update(self, **kwargs):
#         '''
#         Updates the CLF-CBF pair.
#         '''
#         for key in kwargs:
#             if key == "clf":
#                 self.clf = kwargs[key]
#             if key == "cbf":
#                 self.cbf = kwargs[key]

#         self.Hv = self.clf.get_hessian()
#         self.x0 = self.clf.get_critical()
#         self.Hh = self.cbf.get_hessian()
#         self.p0 = self.cbf.get_critical()
#         self.v0 = self.Hv @ ( self.p0 - self.x0 )

#         self.pencil = LinearMatrixPencil( self.cbf.get_hessian(), self.clf.get_hessian() )
#         self.dim = self.pencil.dim

#         self.compute_q()
#         self.compute_equilibrium()
#         # self.compute_equilibrium2()
#         self.compute_critical()

#     def compute_equilibrium2(self):
#         '''
#         Compute the equilibrium points using new method.
#         '''
#         temp_P = -(self.Hv @ self.x0).reshape(self.dim,1)
#         P_matrix = np.block([ [ self.Hv  , temp_P                        ],
#                               [ temp_P.T , self.x0 @ self.Hv @ self.x0 ] ])

#         temp_Q = -(self.Hh @ self.p0).reshape(self.dim,1)
#         Q_matrix = np.block([ [ self.Hh  , temp_Q                        ],
#                               [ temp_Q.T , self.p0 @ self.Hh @ self.p0 ] ])

#         pencil = LinearMatrixPencil( Q_matrix, P_matrix )
#         # print("Eig = " + str(pencil.eigenvectors))

#         # self.equilibrium_points2 = np.zeros([self.dim, self.dim+1])1
#         self.equilibrium_points2 = []
#         for k in range(np.shape(pencil.eigenvectors)[1]):
#             # if np.abs(pencil.eigenvectors[-1,k]) > 0.0001:
#             # print(pencil.eigenvectors)
#             self.equilibrium_points2.append( (pencil.eigenvectors[0:-1,k]/pencil.eigenvectors[-1,k]).tolist() )

#         self.equilibrium_points2 = np.array(self.equilibrium_points2).T

#         # print("Lambda 1 = " + str(pencil.lambda1))
#         # print("Lambda 2 = " + str(pencil.lambda2))
#         # print("Eq = " + str(self.equilibrium_points2))

#     def compute_q(self):
#         '''
#         This method computes the q-function for the pair.
#         '''
#         # Compute denominator of q
#         pencil_eig = self.pencil.eigenvalues
#         pencil_char = self.pencil.characteristic_poly
#         den_poly = np.polynomial.polynomial.polymul(pencil_char, pencil_char)

#         detHv = np.linalg.det(self.Hv)
#         try:
#             Hv_inv = np.linalg.inv(self.Hv)
#             Hv_adj = detHv*Hv_inv
#         except np.linalg.LinAlgError as error:
#             print(error)
#             return

#         # This computes the pencil adjugate expansion and the set of numerator vectors by the adapted Faddeev-LeVerrier algorithm.
#         D = np.zeros([self.dim, self.dim, self.dim])
#         D[:][:][0] = pow(-1,self.dim-1) * Hv_adj

#         Omega = np.zeros( [ self.dim, self.dim ] )
#         Omega[0,:] = D[:][:][0].dot(self.v0)
#         for k in range(1,self.dim):
#             D[:][:][k] = np.matmul( Hv_inv, np.matmul(self.Hh, D[:][:][k-1]) - pencil_char[k]*np.eye(self.dim) )
#             Omega[k,:] = D[:][:][k].dot(self.v0)

#         # Computes the numerator polynomial
#         W = np.zeros( [ self.dim, self.dim ] )
#         for i in range(self.dim):
#             for j in range(self.dim):
#                 W[i,j] = np.inner(self.Hh.dot(Omega[i,:]), Omega[j,:])

#         num_poly = np.polynomial.polynomial.polyzero
#         for k in range(self.dim):
#             poly_term = np.polynomial.polynomial.polymul( W[:,k], np.eye(self.dim)[:,k] )
#             num_poly = np.polynomial.polynomial.polyadd(num_poly, poly_term)

#         residues, poles, k = signal.residue( np.flip(num_poly), np.flip(den_poly), tol=0.001, rtype='avg' )

#         index = np.argwhere(np.real(residues) < 0.0000001)
#         residues = np.real(np.delete(residues, index))

#         # Computes polynomial roots
#         fzeros = np.real( np.polynomial.polynomial.polyroots(num_poly) )

#         # Filters repeated poles from pencil_eig and numerator_roots
#         repeated_poles = []
#         for i in range( len(pencil_eig) ):
#             for j in range( len(fzeros) ):
#                 if np.absolute(fzeros[j] - pencil_eig[i]) < self.eigen_threshold:
#                     if np.any(repeated_poles == pencil_eig[i]):
#                             break
#                     else:
#                         repeated_poles.append( pencil_eig[i] )
#         repeated_poles = np.array( repeated_poles )

#         self.q_function = {
#                             "denominator": den_poly,
#                             "numerator": num_poly,
#                             "poles": pencil_eig,
#                             "zeros": fzeros,
#                             "repeated_poles": repeated_poles,
#                             "residues": residues }

#     def compute_equilibrium(self):
#         '''
#         Compute equilibrium solutions and equilibrium points.
#         '''
#         solution_poly = np.polynomial.polynomial.polysub( self.q_function["numerator"], self.q_function["denominator"] )

#         equilibrium_solutions = np.polynomial.polynomial.polyroots(solution_poly)
#         equilibrium_solutions = np.real(np.extract( equilibrium_solutions.imag == 0.0, equilibrium_solutions ))
#         equilibrium_solutions = np.concatenate((equilibrium_solutions, self.q_function["repeated_poles"]))

#         # Extract positive solutions and sort array
#         equilibrium_solutions = np.sort( np.extract( equilibrium_solutions > 0, equilibrium_solutions ) )

#         # Compute equilibrium points from equilibrium solutions
#         self.equilibrium_points = np.zeros([self.dim,len(equilibrium_solutions)])
#         for k in range(len(equilibrium_solutions)):
#             if all(np.absolute(equilibrium_solutions[k] - self.pencil.eigenvalues) > self.eigen_threshold ):
#                 self.equilibrium_points[:,k] = self.v_values( equilibrium_solutions[k] ) + self.p0

#     def compute_critical(self):
#         '''
#         Computes critical points of the q-function.
#         '''
#         dnum_poly = np.polynomial.polynomial.polyder(self.q_function["numerator"])
#         dpencil_char = np.polynomial.polynomial.polyder(self.pencil.characteristic_poly)

#         poly1 = np.polynomial.polynomial.polymul(dnum_poly, self.pencil.characteristic_poly)
#         poly2 = 2*np.polynomial.polynomial.polymul(self.q_function["numerator"], dpencil_char)
#         num_df = np.polynomial.polynomial.polysub( poly1, poly2 )

#         self.q_critical_points = np.polynomial.polynomial.polyroots(num_df)
#         self.q_critical_points = np.real(np.extract( self.q_critical_points.imag == 0.0, self.q_critical_points ))

#         # critical_values = self.q_values(self.q_critical)
#         # number_critical = len(self.critical_values)

#         # # Get positive critical points
#         # index, = np.where(self.q_critical > 0)
#         # positive_q_critical = self.q_critical[index]
#         # positive_critical_values = self.critical_values[index]
#         # num_positive_critical = len(self.positive_q_critical)

#     def q_values(self, args):
#         '''
#         Returns the q-function values at given points.
#         '''
#         numpoints = len(args)
#         qvalues = np.zeros(numpoints)
#         for k in range(numpoints):
#             num_value = np.polynomial.polynomial.polyval( args[k], self.q_function["numerator"] )
#             pencil_char_value = np.polynomial.polynomial.polyval( args[k], self.pencil.characteristic_poly )
#             qvalues[k] = num_value/(pencil_char_value**2)

#         return qvalues

#     def v_values( self, lambda_var ):
#         '''
#         Returns the value of v(lambda) = H(lambda)^{-1} v0
#         '''
#         pencil_inv = np.linalg.inv( self.pencil.value( lambda_var ) )
#         return pencil_inv.dot(self.v0)

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