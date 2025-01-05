from common import *
from .basic import Quadratic
from .multipoly import MultiPoly, Poly
from .kernel import Kernel, KernelQuadratic

from time import perf_counter
from scipy.optimize import minimize, least_squares, LinearConstraint, NonlinearConstraint
from shapely import geometry, intersection

from numpy.polynomial import Polynomial as npPoly
from numpy.polynomial.polynomial import polyder, polyint

import warnings
import itertools
import numpy as np
import scipy as sp
import cvxpy as cp
import contourpy as ctp
import matplotlib.colors as mcolors
import platform, os

def param2H(param: np.ndarray):
    '''
    Converts a parameter vector of dimension n(n+1)/2 into a p.s.d. matrix H.
    '''
    L = vector2sym(param)
    H = L.T @ L
    return H

def H2param(H: np.ndarray):
    '''
    Converts a symmetric p.s.d. matrix H into the flattened version
    of its unique square root ( a vector of dimension n(n+1)/2 ).
    '''
    if np.linalg.norm(H - H.T) > 1e-6:
        raise TypeError("Passed matrix is not symmetric.")

    eigHv = np.linalg.eigvals(H)
    if np.any(eigHv) < 0.0:
        raise TypeError("Passed matrix is not p.s.d.")

    L = sp.linalg.sqrtm(H)
    return sym2vector(L)

def dV_matrix(H: np.ndarray, A: np.ndarray, tol: float = 0):
    ''' 
    Returns the corresponding time derivative matrix of the Lyapunov function 
    V = x H x for the LTI autonomous system dx = A x. 
    If tol was passed, add it as an eigenvalue tolerance. 
    '''
    if A.shape != H.shape:
        raise Exception("Matrices A and H must be of the same shape.")
    dim = A.shape[0]

    if np.any([ eig.real > 0 for eig in np.linalg.eigvals(A) ]):
        raise Exception("Passed A matrix is not Lipshitz.")

    return H @ A + A.T @ H + tol*np.eye(dim)

class QuadraticLyapunov(Quadratic):
    '''
    Class for Quadratic Lyapunov functions of the type (x-x0)'Hv(x-x0), parametrized by vector pi_v.
    Here, the Lyapunov minimum is a constant vector x0, and the hessian Hv is positive definite and parametrized by:
    Hv = Lv(pi_v)'Lv(pi_v) + epsilon I_n (Lv is upper triangular and epsilon is a small positive constant).
    '''
    def __init__(self, *args, **kwargs):

        kwargs["height"] = 0.0
        super().__init__(*args, **kwargs)

        self.tol_CLFcond = 1e-1

    def _init_clf_opt(self, A: np.ndarray):
        ''' Setup convex optimization for CLF condition '''

        self.Hvar = cvx.Variable( shape=(self._dim, self._dim), PSD=True )
        cost = cvx.norm( self.Hvar - self.H ) 
        constraint = [ dV_matrix(self.Hvar, A, self.tol_CLFcond) << 0 ]
        self.clf_cond_prob = cvx.Problem( cvx.Minimize( cost ), constraints=constraint )

    def find_clf(self):
        ''' Returns a new CLF Hessian satisfying the CLF condition '''

        self.clf_cond_prob.solve(solver = 'SCS', verbose=False)
        return self.Hvar.value

    def satisfy_clf(self, A):
        ''' 
        Test if QuadraticLyapunov satisfies the CLF condition for LTI system with state matrix A.
        If not, computes a new valid CLF.
        '''
        if np.all(A == np.zeros(A.shape)):              # ignore if system is driftless
            return True 

        self._init_clf_opt(A)
        Hv = self.H
        Q = dV_matrix(Hv, A, self.tol_CLFcond)
        eigsQ = np.linalg.eigvals(Q)
        is_clf = np.all(eigsQ < 0)

        if is_clf:
            print(f"CLF satisfies the CLF condition.")
        else:
            newHv = self.find_clf()
            self.set_params(hessian = newHv)
            print(f"CLF does not satisfy the CLF condition. Closest valid CLF = \n{self.H}")

        return is_clf

    def partial_Hv(self):
        ''' Returns the partial derivatives of Hv wrt to the parameters '''

        sym_basis = symmetric_basis(self._dim)
        param = H2param( self.H )

        partial_Hv = np.zeros([ len(param), self._dim, self._dim ])
        for i,j in zip(range(len(param)), range(len(param))):
            partial_Hv[i,:,:] = partial_Hv[i,:,:] + ( sym_basis[i].T @ sym_basis[j] + sym_basis[j].T @ sym_basis[i] )*param[j]

        return partial_Hv

    def gamma_transform(self, x: np.ndarray, gamma: npPoly):
        ''' 
        Using a K infinity gamma polynomial, 
        returns the function, gradient and Hessian values of 
        the integral transformation.
        '''
        if not isinstance(gamma, npPoly):
            raise TypeError("Gamma function is not a polynomial.")
        
        gamma_der = npPoly( polyder(gamma.coef) )
        gamma_int = npPoly( polyint(gamma.coef) )

        V = self(x)
        gradV = self.gradient(x)
        HV = self.hessian(x)

        gammaV = gamma(V)
        gammaV_der = gamma_der(V)
        barV = gamma_int(V)
        bar_gradV = gammaV * gradV
        barHv = gammaV * HV + gammaV_der * np.outer(gradV, gradV)

        return barV, bar_gradV, barHv

    def inverse_gamma_transform(self, x: np.ndarray, gamma: npPoly):
        '''
        Using a K infinity gamma polynomial, 
        returns the function, gradient and Hessian values of 
        the inverse integral transformation.
        '''
        if not isinstance(gamma, npPoly):
            raise TypeError("Gamma function is not a polynomial.")
                
        gamma_der = npPoly( polyder(gamma.coef) )
        gamma_int = npPoly( polyint(gamma.coef) )
        
        # Get transformed V(x), nablaV and Hessian
        barV = self(x)
        bar_gradV = self.gradient(x)
        bar_Hv = self.hessian(x)

        # Compute original V, nablaV and Hessian
        V0 = 100*np.random.rand()
        sol = least_squares( lambda V: barV - gamma_int( V ), V0 )
        V = sol.x[0]

        if sol.fun >= 1e-10:
            raise ValueError("Optimization did not converge.")
        
        gammaV = gamma(V)
        gradV = bar_gradV / gammaV

        gammaderV = gamma_der(V)
        Hv = ( 1/gammaV ) * ( bar_Hv - gammaderV * np.outer( gradV,gradV ) )

        return V, gradV, Hv

    @classmethod
    def geometry(cls, semiaxes: tuple, R: np.ndarray, center: list | np.ndarray, **kwargs):
        ''' Create QuadraticLyapunov from geometric parameters: semiaxes lengths, rotation matrix R and center '''

        eigs = [ 1/(ax**2) for ax in semiaxes ]
        H = R @ np.diag(eigs) @ R.T
        return cls(hessian=H, center=center, height=0.0, **kwargs)

    @classmethod
    def geometry2D(cls, semiaxes: tuple, angle: float, center: list | np.ndarray, **kwargs):
        ''' Create QuadraticLyapunov from 2D geometric parameters: semiaxes lengths, angle and center '''
        R = rot2D(angle)   
        return QuadraticLyapunov.geometry(semiaxes, R, center, **kwargs)

class QuadraticBarrier(Quadratic):
    '''
    Class for Quadratic barrier functions. For positive definite Hessians, the unsafe set is described by the interior of an ellipsoid.
    The symmetric Hessian is parametrized by Hh(pi) = sum^n_i Li pi_i, where {Li} is the canonical basis of the space of (n,n) symmetric matrices.
    '''
    def __init__(self, *args, **kwargs):

        kwargs["height"] = -0.5
        super().__init__(*args, **kwargs)

    @classmethod
    def geometry(cls, semiaxes: tuple, R: np.ndarray, center: list | np.ndarray, **kwargs):
        ''' Create Quadratic from geometric parameters: semiaxes lengths, rotation matrix R and center '''

        eigs = [ 1/(ax**2) for ax in semiaxes ]
        H = R @ np.diag(eigs) @ R.T
        return cls(hessian=H, center=center, height=-0.5, **kwargs)

    @classmethod
    def geometry2D(cls, semiaxes: tuple, angle: float, center: list | np.ndarray, **kwargs):
        ''' Create QuadraticBarrier from 2D geometric parameters: semiaxes lengths, angle and center '''

        R = rot2D(angle)    
        return QuadraticBarrier.geometry(semiaxes, R, center, **kwargs)

class KernelLyapunov(KernelQuadratic):
    '''
    Class for kernel-based Lyapunov functions.
    Derived from KernelQuadratic, comprises CLF V(x) = ½ m(x)' P m(x),
    where P is the shape matrix.
    Receives a polynomial class K gamma function
    '''
    def __init__(self, **kwargs):

        kwargs["color"] = mcolors.TABLEAU_COLORS['tab:blue'] # Standard CLF color is blue; however, this can be overwritten
        kwargs["constant"] = 0.0                             # Standard constant for CLF is 0.0 (cannot be overwritten)

        # self.gamma_fun = Poly(kwargs["gamma"])
        # self.gamma_fun_int = np.polyint(self.gamma_fun)
        # self.gamma_fun_der = np.polyder(self.gamma_fun)

        # Initialize the parameters of KernelQuadratic
        super().__init__(**kwargs)

    def __str__(self):
        return "Polynominal kernel-based CLF V(x) = ½ m(x)' P m(x)"

    def _fun(self, x, shape_matrix):

        if isinstance(shape_matrix, cp.expressions.variable.Variable):
            return super()._fun(x, shape_matrix)

        barV = super()._fun(x, shape_matrix)
        # return np.sqrt( 2 * barV )
        return barV

    def _grad(self, x, shape_matrix):

        if isinstance(shape_matrix, cp.expressions.variable.Variable):
            return super()._grad(x, shape_matrix)

        # V = self._fun(x, shape_matrix)
        grad_barV = super()._grad(x, shape_matrix)
        # return (1/V)*grad_barV
        return grad_barV

    def _hess(self, x, shape_matrix):

        if isinstance(shape_matrix, cp.expressions.variable.Variable):
            return super()._hess(x, shape_matrix)

        # V = self._fun(x, shape_matrix)
        # gradV = self._grad(x, shape_matrix)
        hessian_barV = super()._hess(x, shape_matrix)
        # return (1/V)*( hessian_barV - np.outer(gradV, gradV) )
        return hessian_barV

    def _function(self, point: np.ndarray) -> np.ndarray:
        '''
        Computes FUNCTION V = ½ m(x)' P m(x)
        '''
        return self._fun(self._validate(point), self.P)

    def _gradient(self, point: np.ndarray) -> np.ndarray:
        '''
        Computes GRADIENT ∇V = Jm(x)' P m(x)
        '''
        return self._grad(self._validate(point), self.P)

    def _hessian(self, point: np.ndarray) -> np.ndarray:
        '''
        Computes HESSIAN ∇x∇(V)
        '''
        return self._hess(self._validate(point), self.P)

    def function_from_P(self, x, P):
        ''' Computes FUNCTION V = ½ m(x)' P m(x) '''
        return self._fun(x, P)

    def gradient_from_P(self, x, P):
        ''' Computes GRADIENT ∇V = Jm(x)' P m(x) '''
        return self._grad(x, P)

    def hessian_from_P(self, x, P):
        ''' Computes HESSIAN ½ V**2 = ½ k(x)' P k(x) '''
        return self._hess(x, P)

    def set_params(self, **kwargs):
        '''
        Set the parameters of the Kernel Lyapunov function.
        Optional: pass a vector of parameters representing the vectorization of matrix P
        '''
        if "P" in kwargs.keys():
            kwargs["coefficients"] = kwargs.pop("P") # Standard name for CLF shape matrix is P

        super().set_params(**kwargs)
        self.P = self.shape_matrix

    def gamma_transform(self, x: np.ndarray, gamma: Poly):
        ''' 
        Using a K infinity gamma polynomial, computes the values of 
        barV(X), ∇barV(x) and H_barV(x)
        '''
        if not isinstance(gamma, Poly):
            raise TypeError("Gamma function is not a polynomial.")
        
        gamma_der = Poly( polyder(gamma.coef) )
        gamma_int = Poly( polyint(gamma.coef) )

        V = self.function(x)
        gradV = self.gradient(x)
        HV = self.hessian(x)

        gammaV = gamma(V)
        gammaV_der = gamma_der(V)
        barV = gamma_int(V)
        bar_gradV = gammaV * gradV
        barHv = gammaV * HV + gammaV_der * np.outer(gradV, gradV)

        return barV, bar_gradV, barHv

class KernelBarrier(KernelQuadratic):
    '''
    Class for kernel-based barrier functions.
    '''
    def __init__(self, **kwargs):

        kwargs["color"] = mcolors.TABLEAU_COLORS['tab:red'] # Standard CBF color is red; however, this can be overwritten
        kwargs["constant"] = 0.5                            # Standard constant for CBF is 0.5 (cannot be overwritten)

        super().__init__(**kwargs)

    def __str__(self):
        return "Polynominal kernel-based CBF h(x) = ½ ( k(x)' Q k(x) - 1 )"

    def set_params(self, **kwargs):
        ''' Set the parameters of the Kernel Barrier function.
        Optional: pass a vector of parameters representing the vectorization of matrix Q '''

        if "Q" in kwargs.keys():
            kwargs["coefficients"] = kwargs.pop("Q") # Standard name for CBF shape matrix is Q

        super().set_params(**kwargs)
        self.Q = self.shape_matrix

    def get_boundary(self):
        ''' Computes the boundary level set '''
        return self.get_levels(levels=[0.0])[0]

class LyapunovBarrier():
    '''
    Class for safety-critical control algorithms with polynomial-based plant, CLF and CBFs (any number of them).
    Defines common algorithms for CLF-CBF pair compatibility, such as:
      (i) computation of the invariat set, 
     (ii) equilibrium points,
    (iii) optimizations over the invariant set branches.
    The variable self.P is used for online computations with the CLF shape.
    '''
    def __init__(self, **kwargs):

        self.plant = None
        self.clf = None
        self.tclf = None
        self.cbfs = []

        # Default QP parameters. Gamma and alpha functions are identity polynomials
        self.params = { "slack_gain": 1.0, "gamma": Poly([0.0, 1.0]), "alpha": Poly([0.0, 1.0]) }
        self.params["dgamma"] = Poly( polyder(self.params["gamma"].coef) )
        self.params["intgamma"] = Poly( polyint(self.params["gamma"].coef) )

        # Invariant set computation and equilibria parameters
        self.limits = [ [-1, +1] for _ in range(2) ]
        self.spacing = 0.1
        self.invariant_color = mcolors.BASE_COLORS["k"]
        self.invariant_complete = False
        self.interior_eq_threshold = 1e-1
        self.max_P_eig = 100
        self.plotted_attrs = {}

        # Dict of compatibilization process parameters
        self.comp_process_params = { "stability_threshold": 0.1,
                                     "measure_threshold": 0.01,
                                     "G": hessian_2Dquadratic(eigen=[1, 1], angle=np.deg2rad(45)),
                                     "invex_tol": 1e-0,
                                     "max_P_eigenvalue": 10 }

        # Dict for storing compatibilization process data
        self.comp_process_data = {"step": 0,
                                  "start_time": 0.0,
                                  "execution_time": 0.0,
                                  "gui_eventloop_time": 0.3 }

        # Dict for storing compatibilization graphics data
        self.comp_graphics = { "fig": None,
                               "text": None,
                               "clf_artists": [] }

        # Defines main adjustable class attributes
        self.set_param(**kwargs)

        # Initialize cost and constraints for scipy.optimize computation
        # self.cost = 0.0
        # self.invexity = 0.0
        # self.centering = 0.0
        # self.max_eig_constr = 0.0
        # self.counter = 0

        # tol = self.comp_process_params["invex_tol"]
        # dim_D = self.kernel.dim_det_kernel
        # self.Tol = np.zeros((dim_D,dim_D))
        # self.Tol[0,0] = 1
        # self.Tol = tol*self.Tol

        # self.stability_pressures = np.zeros(self.num_cbfs)
        # self.measures = np.zeros(self.num_cbfs)

    def compute_cbf_boundary(self, cbf_index):
        '''Compute CBF boundary'''

        # unsafe_set = geometry.Polygon()
        for boundary_seg in self.cbfs[cbf_index].get_boundary():
            boundary_lines = geometry.LineString(boundary_seg)
            self.boundary_lines[cbf_index].append(boundary_lines)

            # unsafe_set = unsafe_set | geometry.Polygon(boundary_seg)

        # self.safe_sets[cbf_index] = self.world_polygon.difference(unsafe_set)
        # self.unsafe_sets[cbf_index] = unsafe_set

    def set_param(self, **kwargs):
        '''
        Sets the following parameters: limits, spacing (for invariant set and equilibria computation)
                                       invariant_color, equilibria_color
                                       plant, clf, cbf, params
        '''
        update_invariant = False

        from dynamic_systems import PolyAffineSystem

        if "limits" in kwargs.keys():
            self.limits = kwargs["limits"]
            xmin, xmax, ymin, ymax = self.limits
            world_coords = [ (xmax, ymax), (xmax, ymin), (xmin, ymin), (xmin, ymax) ]
            self.world_polygon = geometry.Polygon(world_coords)

        if "spacing" in kwargs.keys():
            self.spacing = kwargs["spacing"]

        for key in kwargs.keys():

            if key in ("limits", "spacing"): # already dealt with
                continue

            if key == "invariant_color":
                self.invariant_color = kwargs["invariant_color"]
                continue

            if key == "equilibria_color":
                self.equilibria_color = kwargs["equilibria_color"]
                continue

            if key == "plant":
                if not isinstance(kwargs["plant"], PolyAffineSystem):
                    raise Exception("Plant is not a polynomial affine system.")
                self.plant = kwargs["plant"]
                update_invariant = True
                continue

            if key == "clf":
                if not isinstance(kwargs["clf"], KernelLyapunov):
                    raise Exception("CLF is not a Kernel polynomial function.")
                self.clf = kwargs["clf"]
                update_invariant = True
                continue

            if key == "tclf":
                if not isinstance(kwargs["tclf"], MultiPoly):
                    raise Exception("Transformed CLF is not a polynomial.")
                
                self.tclf = kwargs["tclf"]

                if self.tclf._poly_grad is None:
                        self.tclf.poly_grad()
                self.grad_tclf = self.tclf._poly_grad

                if self.tclf._poly_hess is None:
                        self.tclf.poly_hess()
                self.hess_tclf = self.tclf._poly_hess

                update_invariant = True
                continue

            if key == "cbfs":
                for cbf in kwargs["cbfs"]:
                    if not isinstance(cbf, KernelBarrier):
                        raise Exception("CBF is not a Kernel polynomial function.")

                self.cbfs: list[Poly] = kwargs["cbfs"]
                self.num_cbfs = len(self.cbfs)
                self.boundary_lines = [ [] for _ in self.cbfs ]
                self.safe_sets = [ geometry.Polygon() for _ in self.cbfs ]
                self.unsafe_sets = [ geometry.Polygon() for _ in self.cbfs ]
                self.invariant_lines_plot = [ [] for _ in self.cbfs ]
                self.comp_process_data["invariant_segs_log"] = [ [] for _ in self.cbfs ]
                for cbf_index in range(self.num_cbfs):
                    self.compute_cbf_boundary(cbf_index)
                update_invariant = True
                continue

            if key == "params":
                params = kwargs["params"]
                for key in params.keys():
                    if key == "slack_gain":
                        self.params["slack_gain"] = params["slack_gain"]
                        update_invariant = True
                        continue
                    if key == "gamma":
                        if not isinstance(params["gamma"], Poly):
                            raise Exception("Passed gamma is not a numpy polynomial.")
                        self.params["gamma"] = params["gamma"]
                        self.params["dgamma"] = Poly( polyder(self.params["gamma"].coef) )
                        self.params["intgamma"] = Poly( polyint(self.params["gamma"].coef) )
                        update_invariant = True
                        continue
                    if key == "alpha":
                        if not isinstance(params["alpha"], Poly):
                            raise Exception("Passed alpha is not a numpy polynomial.")
                        self.params["alpha"] = params["alpha"]
                        update_invariant = True
                        continue
                continue

        if ( self.clf is not None ) and (self.tclf is None):

            clf_poly = self.clf.to_multipoly()
            grad_clf_poly = clf_poly.poly_grad()
            hess_clf_poly = clf_poly.poly_hess()

            gamma = self.params["gamma"]
            dgamma = self.params["dgamma"]
            intgamma = self.params["intgamma"]

            gammaV_poly = gamma(clf_poly)
            self.tclf = intgamma(clf_poly)
            self.grad_tclf = gammaV_poly * grad_clf_poly
            self.hess_tclf = gammaV_poly * hess_clf_poly + dgamma(clf_poly) * MultiPoly.outer( grad_clf_poly, grad_clf_poly )

        # Initialize grids used for invariant set computation
        if hasattr(self, "limits") and hasattr(self, "spacing"):
            xmin, xmax, ymin, ymax = self.limits
            x = np.arange(xmin, xmax, self.spacing)
            y = np.arange(ymin, ymax, self.spacing)

            self.xg, self.yg = np.meshgrid(x,y)
            self.grid_shape = self.xg.shape
            self.grid_pts = list( zip( self.xg.flatten(), self.yg.flatten() ) )
            self.determinant_grid = [ np.empty(self.grid_shape, dtype=float) for _ in self.cbfs ]
            self.area_function = [ np.empty(self.grid_shape, dtype=float) for _ in self.cbfs ]

        self.verify_kernels()
        if update_invariant:
            self.update_invariant_set(verbose=True)

    def verify_kernels(self):
        '''
        Verifies if the kernel pair is consistent and fully defined
        '''
        equal_kernel_dims = [ self.plant.f_poly.n == self.plant.g_poly.n ]

        if ( self.clf is None ) and (self.tclf is not None):
            equal_kernel_dims += [ self.plant.g_poly.n == self.tclf.n ]
            equal_kernel_dims += [ self.tclf.n == cbf.kernel._dim for cbf in self.cbfs ]
        else:
            equal_kernel_dims += [ self.plant.g_poly.n == self.clf.kernel._dim ]
            equal_kernel_dims += [ self.clf.kernel._dim == cbf.kernel._dim for cbf in self.cbfs ]
            self.P = self.clf.P

        if not all(equal_kernel_dims):
            raise Exception("Plant, CLF and CBF do not have the same dimension.")

        self.n = self.plant.f_poly.n
        # self.p = self.plant.kernel._num_monomials
        
    def invariant_pencil(self, cbf_index: int = 0):
        '''
        Computes the invariant equation
        f(x) + λ G(x) ∇h(x) - p γ(V(x)) G(x) ∇V(x) = (λ A - B) m(x)
        in singular pencil format, that is, the matrix pencil (A, B) with a corresponding kernel function m(x).
        '''
        p = self.params["slack_gain"]

        f_poly = self.plant.get_fpoly()
        G_poly = self.plant.get_Gpoly()

        # Get CBF polynomials
        h_poly = self.cbfs[cbf_index].to_multipoly()
        nablah_poly = h_poly.poly_grad()

        vecQ_poly = G_poly @ nablah_poly
        vecP_poly = p * ( G_poly @ self.grad_tclf ) - f_poly

        vecQ_poly.filter()
        vecP_poly.filter()

        A_poly = vecQ_poly + 0 * vecP_poly
        B_poly = 0 * vecQ_poly + vecP_poly

        A_poly.sort_kernel()
        B_poly.sort_kernel()

        if A_poly.kernel != B_poly.kernel:
            raise Exception("Kernels are not the same. This should not happen.")

        A = np.array([ cA.tolist() for cA in A_poly.coeffs ]).T
        B = np.array([ cB.tolist() for cB in B_poly.coeffs ]).T

        return A, B, A_poly.kernel

    def vecQ_fun(self, pt: np.ndarray, cbf_index: int) -> np.ndarray:
        '''
        Returns the vector vQ = G ∇h
        '''
        G = self.plant.get_G(pt)
        nablah = self.cbfs[cbf_index].gradient(pt)
        return G @ nablah

    def vecP_fun(self, pt: np.ndarray) -> np.ndarray:
        '''
        Returns the vector vP = p γ(V(x,P)) G(x) ∇V(x) - f(x) with CLF shape matrix P
        '''
        p = self.params["slack_gain"]

        f = self.plant.get_f(pt)
        G = self.plant.get_G(pt)

        return p * ( G @ self.grad_tclf(pt) ) - f

    def clf_fun(self, pt: np.ndarray) -> float:
        '''
        If a tCLF is directly specified, returns the computed CLF value from tclf and inverse_gamma_transform.
        If only a CLF is specified, returns the CLF value using self P matrix.
        '''
        if (self.tclf is not None):
            V, gradV, hessV = self.tclf.inverse_gamma_transform(pt, self.params["gamma"])
            return V
        else:
            return self.clf.function_from_P(pt, self.P)

    def clf_gradient(self, pt: np.ndarray) -> np.ndarray:
        '''
        If a tCLF is directly specified, returns the computed CLF gradient from tclf and inverse_gamma_transform.
        If only a CLF is specified, returns the CLF gradient using self P matrix.
        '''
        if (self.tclf is not None):
            V, gradV, hessV = self.tclf.inverse_gamma_transform(pt, self.params["gamma"])
            return gradV
        else:
            return np.array( self.clf.gradient_from_P(pt, self.P) )

    def clf_hessian(self, pt: np.ndarray) -> np.ndarray:
        '''
        If a tCLF is directly specified, returns the computed CLF Hessian from tclf and inverse_gamma_transform.
        If only a CLF is specified, returns the CLF Hessian using self P matrix.
        '''
        if (self.tclf is not None):
            V, gradV, hessV = self.tclf.inverse_gamma_transform(pt, self.params["gamma"])
            return hessV
        else:
            return np.array( self.clf.hessian_from_P(pt, self.P) )

    def lambda_fun(self, pt: np.ndarray, cbf_index: int) -> float:
        '''
        Given a point x in an invariant set, compute its corresponding lambda scalar.
        '''
        # nablah = self.cbfs[cbf_index].gradient(pt)
        vQ = self.vecQ_fun(pt, cbf_index)
        vP = self.vecP_fun(pt)
        # return (nablah.T @ vP) / ( nablah.T @ vQ )

        return (vQ.T @ vP) / ( vQ.T @ vQ )

    def invariant_Jacobian(self, pt: np.ndarray, cbf_index: int = -1) -> np.ndarray:
        ''' Computes the invariant Jacobian to determine the stability of equilibrium points '''

        G = self.plant.get_G(pt)
        Jf = self.plant.get_Jf(pt)
        JG_list = self.plant.get_JG_list(pt)

        grad_barV = self.grad_tclf(pt)
        hess_barV = self.hess_tclf(pt)

        if cbf_index >= 0:
            nablah = self.cbfs[cbf_index].gradient(pt)
            Hh = self.cbfs[cbf_index].hessian(pt)
            l = self.lambda_fun(pt, cbf_index)
        else:
            nablah = np.zeros(self.n)
            Hh = np.zeros((self.n,self.n))
            l = 0.0

        p = self.params["slack_gain"]
        pencil = l * nablah - p * grad_barV
        Hpencil = l * Hh - p * hess_barV
        JG_term = np.array([ JG @ pencil for JG in JG_list ])

        Jfi = Jf + JG_term + G @ Hpencil

        return Jfi

    def stability_fun(self, x_eq, type_eq, cbf_index: int = -1):
        '''
        Compute the stability number for a given equilibrium point
        '''
        V = self.clf_fun(x_eq)
        nablaV = self.clf_gradient(x_eq)

        if cbf_index >= 0:
            nablah = self.cbfs[cbf_index].gradient(x_eq)
            norm_nablaV = np.linalg.norm(nablaV)
            norm_nablah = np.linalg.norm(nablah)
            unit_nablah = nablah/norm_nablah

        Jfi = self.invariant_Jacobian(x_eq, cbf_index)

        if type_eq == "boundary":
            curvatures, basis_for_TpS = compute_curvatures( Jfi, unit_nablah )
            max_index = np.argmax(curvatures)
            # stability_number = curvatures[max_index] / ( self.params["slack_gain"] * self.params["clf_gain"] * norm_nablaV )
            stability_number = curvatures[max_index] / norm_nablaV

        if type_eq == "interior":
            stability_number = np.max( np.linalg.eigvals( Jfi ) )

        '''
        If the CLF-CBF gradients are collinear, then the stability_number is equivalent to the diff. btw CBF and CLF curvatures at the equilibrium point:
        '''
        eta = self.eta_fun(x_eq, cbf_index)
        return stability_number, eta

    def eta_fun(self, x_eq, cbf_index):
        '''
        Returns the value of eta (between 0 and 1), depending on the collinearity between the CLF-CBF gradients.
        '''
        if cbf_index < 0: return

        gradV = self.clf_gradient(x_eq)
        gradh = self.cbfs[cbf_index].gradient(x_eq)

        p = self.params["slack_gain"]
        G = self.plant.get_G(x_eq)
        z1 = gradh / np.sqrt(gradh.T @ G @ gradh)
        z2 = gradV - gradV.T @ G @ z1 * z1
        eta = 1/(1 + p * z2.T @ G @ z2 )
        return eta

    ''' ----------------------------------------EQUILIBRIUM ANALYSIS CODE ----------------------------------------------- '''
    def update_determinant_grid(self):
        '''
        Evaluates det([ vQ, vP ]) = 0 over a grid for each CBF.
        Returns: a grid of shape self.grid_shape with the determinant values corresponding to each pt on the grid
        '''
        W_list = [ [] for _ in self.cbfs ]
        lambda_grid = [ [] for _ in self.cbfs ]
        barrier_grid = [ [] for _ in self.cbfs ]

        for pt in self.grid_pts:

            vP = self.vecP_fun(pt).reshape(-1,1)
            for cbf_index in range(self.num_cbfs):
                vQ = self.vecQ_fun(pt, cbf_index).reshape(-1,1)

                W = np.hstack([vQ, vP])
                W_list[cbf_index].append(W)

                l = self.lambda_fun(pt, cbf_index)
                lambda_grid[cbf_index].append( l )

                h = self.cbfs[cbf_index].function(pt)
                barrier_grid[cbf_index].append( h )

        for cbf_index in range(self.num_cbfs):

            determinant_list = np.linalg.det( W_list[cbf_index] )

            # Eliminate the negative lambda part
            if not self.invariant_complete:
                for k, l in enumerate(lambda_grid[cbf_index]):
                    if l < 0.0:
                        determinant_list[k] = np.inf

            self.determinant_grid[cbf_index] = determinant_list.reshape(self.grid_shape)
            self.area_function[cbf_index] = ( determinant_list * np.array( barrier_grid[cbf_index] ) ).reshape(self.grid_shape)

    def update_invariant_set(self, verbose=False):
        '''
        Computes the invariant sets.
        '''
        if self.n > 2:
            warnings.warn("Currently, the computation of the invariant set is not available for dimensions higher than 2.")
            return

        self.update_determinant_grid()                                       # updates the grid with new determinant values

        self.boundary_equilibria, self.interior_equilibria = [], []
        self.stable_equilibria, self.unstable_equilibria = [], []

        self.invariant_lines = [ None for _ in self.cbfs ]
        self.invariant_segs = [ [] for _ in self.cbfs ]
        for cbf_index in range(self.num_cbfs):
            self.invariant_set_analysis(cbf_index)                           # run through each branch of the invariant set of the given CBF

        if verbose:
            show_message(self.boundary_equilibria, "boundary equilibrium points")
            show_message(self.interior_equilibria, "interior equilibrium points")

    def invariant_set_analysis(self, cbf_index: int):
        '''
        Populates invariant segments with data and compute equilibrium points from invariant line data
        corresponding to a given CBF.
        '''
        invariant_contour = ctp.contour_generator( x=self.xg, y=self.yg, z=self.determinant_grid[cbf_index] )  # creates new contour_generator object
        self.invariant_lines[cbf_index] = invariant_contour.lines(0.0)                                         # returns the 0-valued contour lines

        p = self.params["slack_gain"]
        gamma = self.params["gamma"]

        boundary_eqs, interior_eqs = [], []
        stable_eqs, unstable_eqs = [], []
        for segment_points in self.invariant_lines[cbf_index]:

            # ----- Loads segment dictionary
            seg_dict = {"points": segment_points}
            seg_dict["geom"] = geometry.LineString(segment_points)

            seg_dict["lambdas"] = [ self.lambda_fun(pt, cbf_index) for pt in segment_points ]

            first_pt = segment_points[0]
            last_pt = segment_points[-1]
            seg_dict["extremal_pairs"] = {"first": ( self.lambda_fun(first_pt, cbf_index), np.linalg.norm(self.cbfs[cbf_index].gradient(first_pt)) ),
                                          "last": ( self.lambda_fun(last_pt, cbf_index), np.linalg.norm(self.cbfs[cbf_index].gradient(last_pt)) ) }

            seg_dict["clf_values"] = [ self.clf_fun(pt) for pt in segment_points ]
            seg_dict["clf_gradients"] = [ self.clf_gradient(pt) for pt in segment_points ]

            seg_dict["cbf_index"] = cbf_index
            seg_dict["cbf_values"] = [ self.cbfs[cbf_index].function(pt) for pt in segment_points ]

            seg_dict["normalized_lambdas"] = [ l / ( p * gamma(seg_dict["clf_values"][k]) ) for k, l in enumerate( seg_dict["lambdas"] ) ]

            # ----- Computes the corresponding equilibrium points and critical segment values
            self.seg_boundary_equilibria(seg_dict)
            self.seg_interior_equilibria(seg_dict)
            # self.seg_stability_pressure(seg_dict)
            # self.seg_removable_measure(seg_dict)

            # ----- Adds the segment dicts and equilibrium points to corresponding data structures
            self.invariant_segs[cbf_index].append(seg_dict)
            boundary_eqs += seg_dict["boundary_equilibria"]
            interior_eqs += seg_dict["interior_equilibria"]

        for b_eq in boundary_eqs:
            if b_eq["equilibrium"] == "stable": stable_eqs.append(b_eq)
            if b_eq["equilibrium"] == "unstable": unstable_eqs.append(b_eq)

        for i_eq in interior_eqs:
            if i_eq["equilibrium"] == "stable": stable_eqs.append(i_eq)
            if i_eq["equilibrium"] == "unstable": unstable_eqs.append(i_eq)

        self.boundary_equilibria += boundary_eqs
        self.interior_equilibria += interior_eqs
        self.stable_equilibria += stable_eqs
        self.unstable_equilibria += unstable_eqs

    def get_boundary_intersections( self, boundary_lines, seg_lines ):
        ''' Computes the intersections with boundary segments of a particular segment of the invariant set '''

        intersection_pts = []
        for boundary_line in boundary_lines:
            intersections = intersection( boundary_line, seg_lines )

            new_candidates = []
            if not intersections.is_empty:
                if hasattr(intersections, "geoms"):
                    for geo in intersections.geoms:
                        x, y = geo.xy
                        x, y = list(x), list(y)
                        new_candidates += [ [x[k], y[k]] for k in range(len(x)) ]
                else:
                    x, y = intersections.xy
                    x, y = list(x), list(y)
                    new_candidates += [ [x[k], y[k]] for k in range(len(x)) ]

            intersection_pts += new_candidates

        return intersection_pts

    def seg_boundary_equilibria(self, seg_dict: dict[str,]):
        '''Computes boundary equilibrium points for given segment data '''

        cbf_index = seg_dict["cbf_index"]
        intersection_pts = self.get_boundary_intersections( self.boundary_lines[cbf_index], seg_dict["geom"] )

        # Compute all boundary equilibria: they are all intersection points with positive lambda
        seg_dict["boundary_equilibria"] = []
        for pt in intersection_pts:
            lambda_pt = self.lambda_fun(pt, cbf_index)
            if lambda_pt >= 0.0:

                seg_boundary_equilibrium = {"x": pt}
                seg_boundary_equilibrium["cbf_index"] = cbf_index
                seg_boundary_equilibrium["lambda"] = lambda_pt
                seg_boundary_equilibrium["h"] = self.cbfs[cbf_index].function(pt)
                seg_boundary_equilibrium["nablah"] = self.cbfs[cbf_index].gradient(pt).tolist()

                stability, eta = self.stability_fun(pt, "boundary", cbf_index)
                seg_boundary_equilibrium["eta"], seg_boundary_equilibrium["stability"] = eta, stability
                seg_boundary_equilibrium["equilibrium"] = "stable"
                if stability > 0: seg_boundary_equilibrium["equilibrium"] = "unstable"

                seg_dict["boundary_equilibria"].append( seg_boundary_equilibrium )

    def seg_interior_equilibria(self, seg_dict: dict[str,]):
        ''' Computes interior equilibrium points for given segment data '''

        cbf_index = seg_dict["cbf_index"]
        seg_dict["interior_equilibria"] = []
        seg_data = seg_dict["points"]

        p = self.params["slack_gain"]
        gamma = self.params["gamma"]

        # Computes the costs along the whole segment
        costs = []
        for k, (V, nablaV) in enumerate(zip(seg_dict["clf_values"], seg_dict["clf_gradients"])):
            pt = seg_data[k]
            f = self.plant.get_f(pt)
            G = self.plant.get_G(pt)
            costs.append( np.linalg.norm( f - p * gamma(V) * G @ nablaV ) )

        # Finds separate groups of points with costs below a certain threshold... interior equilibria are computed by extracting the argmin of the cost for each group
        for flag, group in itertools.groupby(zip(seg_data, costs), lambda x: x[1] <= self.interior_eq_threshold):
            if flag:
                group = list(group)
                group_pts = [ ele[0] for ele in group ]
                group_costs = [ ele[1] for ele in group ]
                new_eq = group_pts[np.argmin(group_costs)].tolist()
                seg_dict["interior_equilibria"].append({"x": new_eq,
                                                        "cbf_index": cbf_index,
                                                        "lambda": self.lambda_fun(new_eq, cbf_index),
                                                        "h": self.cbfs[cbf_index].function(new_eq),
                                                        "nablah": self.cbfs[cbf_index].gradient(new_eq).tolist()
                                                        })

        # Computes the equilibrium stability
        for eq in seg_dict["interior_equilibria"]:
            stability, eta = self.stability_fun(eq["x"], "boundary", cbf_index)
            eq["eta"], eq["stability"] = eta, stability
            eq["equilibrium"] = "stable"
            if stability > 0:
                eq["equilibrium"] = "unstable"

    def plot_invariant(self, ax, cbf_index, *args):
        '''
        Plots the invariant set segments corresponding to CBF into ax.
        Optional arguments specify the indexes of each invariant segment to be plotted.
        If no optional argument is passed, plots all invariant segments.
        '''
        # Which segments to plot?
        num_segs_to_plot = len(self.invariant_segs[cbf_index])
        segs_to_plot = [ i for i in range(num_segs_to_plot) ]

        if np.any( np.array(args) > len(self.invariant_segs[cbf_index])-1 ):
            print("Invariant segment list index out of range. Plotting all")

        elif len(args) > 0:
            num_segs_to_plot = len(args)
            segs_to_plot = list(args)

        # Adds or removes lines according to the total number of segments to be plotted
        if num_segs_to_plot >= len(self.invariant_lines_plot[cbf_index]):
            for _ in range(num_segs_to_plot - len(self.invariant_lines_plot[cbf_index])):
                line2D, = ax.plot([],[], color=np.random.rand(3), linestyle='dashed', linewidth=1.2 )
                self.invariant_lines_plot[cbf_index].append(line2D)
        else:
            for _ in range(len(self.invariant_lines_plot[cbf_index]) - num_segs_to_plot):
                self.invariant_lines_plot[cbf_index][-1].remove()
                del self.invariant_lines_plot[cbf_index][-1]

        # UP TO HERE: len(self.invariant_lines_plot) == len(segs_to_plot)

        # Updates segment lines with data from each invariant segment
        for k in range(num_segs_to_plot):
            seg_index = segs_to_plot[k]
            x_seg_points = self.invariant_segs[cbf_index][seg_index]["points"][:,0]
            y_seg_points = self.invariant_segs[cbf_index][seg_index]["points"][:,1]
            self.invariant_lines_plot[cbf_index][k].set_data( x_seg_points, y_seg_points )

    def plot_attr(self, ax, attr_name: str, plot_color='k', alpha=1.0):
        '''
        Plots a list attribute from the class into ax.
        '''
        attr = getattr(self, attr_name)
        if type(attr) != list:
            raise Exception("Passed attribute name is not a list.")

        # If the passed attribute name was not already initialized, initialize it.
        if attr_name not in self.plotted_attrs.keys():
            self.plotted_attrs[attr_name] = []

        # Balance the number of line2D elements in the array of plotted points
        if len(attr) >= len(self.plotted_attrs[attr_name]):
            for _ in range(len(attr) - len(self.plotted_attrs[attr_name])):
                line2D, = ax.plot([],[], 'o', color=plot_color, alpha=alpha, linewidth=0.6 )
                self.plotted_attrs[attr_name].append(line2D)
        else:
            for _ in range(len(self.plotted_attrs[attr_name]) - len(attr)):
                self.plotted_attrs[attr_name][-1].remove()
                del self.plotted_attrs[attr_name][-1]

        # from this point on, len(attr) = len(self.plotted_attrs[attr_name])
        for k in range(len(attr)):
            self.plotted_attrs[attr_name][k].set_data( [attr[k]["x"][0]], [attr[k]["x"][1]] )

    ''' -------------------------------------------- COMPATIBILIZATION CODE ------------------------------------------------- '''
    # def seg_stability_pressure(self, seg_dict: dict[str,]):
    #     '''
    #     Computes the segment stability pressure, expressed as a line integral
    #     '''
    #     cbf_index = seg_dict["cbf_index"]
    #     seg_data = seg_dict["points"]

    #     # Integrate the segment stability pressure
    #     pressure_integral = 0.0
    #     segment_length = 0.0
    #     seg_dict["s"] = [0.0]
    #     for k, pt in enumerate(seg_data[0:-1,:]):
    #         mean_pt = 0.5*( pt + seg_data[k+1,:] )
    #         ds = np.linalg.norm(mean_pt)
    #         segment_length += ds
    #         seg_dict["s"] += [segment_length]
    #         h = self.cbfs[cbf_index].function(mean_pt)

    #         stability, eta = self.stability_fun(mean_pt, type_eq="boundary", cbf_index=cbf_index)
    #         fun = gaussian(h, sigma = 0.1) * ( stability - self.comp_process_params["stability_threshold"] )
    #         pressure_integral += fun*ds

    #     seg_dict["length"] = seg_dict["s"][-1]
    #     seg_dict["gradient_cbf_values"] = np.gradient(seg_dict["cbf_values"], seg_dict["s"])
    #     seg_dict["stability_pressure"] = pressure_integral/segment_length

    # def seg_removable_measure(self, seg_dict: dict[str,]) -> float:
    #     '''
    #     Computes the segment removable measure, if it exists
    #     '''
    #     barrier_vals = seg_dict["cbf_values"]
    #     grad_barrier_vals = seg_dict["gradient_cbf_values"]

    #     # segment is not removable until proven otherwise
    #     seg_dict["removable_measure"] = +np.inf
    #     if len(seg_dict["boundary_equilibria"]) == 0:
    #         return

    #     # If segment starts AND ends completely outside/inside the unsafe set, it's removable
    #     if barrier_vals[0] * barrier_vals[-1] > 0:

    #         # removable from outside
    #         if barrier_vals[0] > 0:
    #             seg_dict["removable_measure"] = min(barrier_vals) - self.comp_process_params["measure_threshold"]

    #         # removable from inside
    #         if barrier_vals[0] < 0:
    #             seg_dict["removable_measure"] = - max(barrier_vals) - self.comp_process_params["measure_threshold"]

    #     # If segment starts/ends inside and ends/starts outside, it's not removable (crosses boundary at least once)
    #     else:

    #         # starts inside (cbf values must be strictly increasing)
    #         if barrier_vals[0] < 0:
    #             seg_dict["removable_measure"] = min(grad_barrier_vals)

    #         # starts outside (cbf values must be strictly decreasing)
    #         if barrier_vals[0] > 0:
    #             seg_dict["removable_measure"] = - max(grad_barrier_vals)

    # def is_compatible(self) -> bool:
    #     '''
    #     Checks if kernel triplet is compatible.
    #     '''
    #     # Checks if boundary has stable equilibria
    #     for boundary_eq in self.boundary_equilibria:
    #         if boundary_eq["equilibrium"] == "stable":
    #             return False

    #     # Checks if more than one stable interior equilibria exist
    #     stable_interior_counter = 0
    #     for interior_eq in self.interior_equilibria:
    #         if interior_eq["equilibrium"] == "stable": stable_interior_counter += 1
    #     if stable_interior_counter > 1: return False

    #     return True

    # def var_to_N(self, var: np.ndarray) -> np.ndarray:
    #     ''' Transforms an n*p array representing a stacked N matrix'''
    #     return var.reshape((self.n, self.p))

    # def N_to_var(self, N: np.ndarray) -> np.ndarray:
    #     '''Transforms matrix N into an array of size n*p'''
    #     return N.flatten()

    # def compute_invex(self, P, center, points=[]):
    #     '''
    #     This method computes the closest invex P to Pinit
    #     '''
    #     G = self.comp_process_params["G"]

    #     Ninit, decomp_cost = NGN_decomposition(G, P)
    #     print(f"Decomposition cost = {decomp_cost}")

    #     # Finds closest invex N to Ninit
    #     Dinit = np.array(self.kernel.D(Ninit))
    #     D_eig = np.linalg.eigvals(Dinit)
    #     print(f"D eigs = {D_eig}")

    #     if len(D_eig) > 1:
    #         if np.abs(min(D_eig)) <= np.abs(max(D_eig)):
    #             cone = +1
    #             print(f"D(N) in psd cone")
    #         else:
    #             cone = -1
    #             print(f"D(N) in nsd cone")
    #     else:
    #         if D_eig >= 0:
    #             cone = +1
    #             print(f"D(N) in psd cone")
    #         else:
    #             cone = -1
    #             print(f"D(N) in nsd cone")

    #     def objective(var: np.ndarray) -> float:
    #         ''' Frobenius norm '''

    #         N = self.var_to_N(var)
    #         P = N.T @ G @ N

    #         cost = 0.0
    #         for pt in points:

    #             # Add clf value
    #             if "level" in pt.keys():
    #                 clf_value = self.clf._fun(pt["coords"], P)
    #                 cost += ( clf_value - pt["level"] )**2

    #             # Add clf gradient
    #             if "gradient" in pt.keys():
    #                 gradient = np.array(pt["gradient"])
    #                 normalized = gradient/np.linalg.norm(gradient)
    #                 clf_grad = self.clf._grad(pt["coords"], P)
    #                 clf_grad_norm = np.linalg.norm(clf_grad)
    #                 cost += ( clf_grad.T @ normalized - clf_grad_norm )**2

    #             # Add clf curvature
    #             if "curvature" in pt.keys():
    #                 if "gradient" not in pt.keys():
    #                     raise Exception("Cannot specify a curvature without specifying the gradient.")
    #                 v = rot2D(np.pi/2) @ normalized
    #                 Hv = self.clf._hess(pt["coords"], P)
    #                 cost += ( v.T @ Hv @ v - pt["curvature"] )**2

    #         Pinit = Ninit.T @ G @ Ninit
    #         return np.linalg.norm( P - Pinit )**2 + cost

    #     def invexity_constr(var: np.ndarray) -> float:
    #         ''' Keeps the CLF invex '''
    #         N = self.var_to_N(var)
    #         D = np.array(self.kernel.D(N))

    #         if cone == +1:
    #             self.invexity = min(np.linalg.eigvals( D - self.Tol ))
    #         if cone == -1:
    #             self.invexity = -max(np.linalg.eigvals( D + self.Tol ))

    #         return self.invexity

    #     def center_constr(var: np.ndarray):
    #         ''' Keeps the CLF centered '''
    #         N = self.var_to_N(var)
    #         m_center = self.kernel.function(center)
    #         P = N.T @ G @ N
    #         return m_center.T @ P @ m_center

    #     constrs = []
    #     constrs += [ {"type": "ineq", "fun": invexity_constr} ]
    #     constrs += [ {"type": "eq", "fun": center_constr} ]

    #     init_var2 = self.N_to_var(Ninit)
    #     sol2 = minimize( objective, init_var2, constraints=constrs, options={"disp": False, "maxiter":1000} )
    #     invex_N =  self.var_to_N( sol2.x )
    #     invex_P = invex_N.T @ G @ invex_N

    #     total_cost = objective(sol2.x)
    #     center_cost = center_constr(sol2.x)
    #     print(f"Total cost = {total_cost}")
    #     print(f"Invexity = {self.invexity}")
    #     print(f"Center cost = {center_cost}")

    #     return invex_P

    # def compatibilize(self, Ninit: np.ndarray, center: np.ndarray, verbose=False, animate=False) -> dict:
    #     '''
    #     This function computes a new CLF geometry that is completely compatible with the original CBF.

    #     The algorithm uses the following ingredients:
    #      1) stability pressure along the invariant sets
    #      2) removable measures between the invariant sets and safe set boundaries
    #      3) parametric invex CLF

    #     Description:

    #     In general, invariant set branches are 1D manifolds of two distinct types:
    #     (i) unbounded branch: goes to infinity from both sides
    #     (ii) bounded branch: ends either in an interior equilibrium point or in ||∇h|| = 0 point; the other side goes → ∞

    #     Equilibrium points can occur in:
    #     (i) stable/unstable pairs, connected through the same invariant set branch;
    #     (ii) singletons, occuring in a bounded branch.

    #     In the case of stable/unstable pairs, the associated branch of the invariant set will have an associated removable area connecting the two equilibria:
    #     namely the area of a minimal surface connecting the interval of the invariant set between the two equilibria with a geodesic on the safe set boundary connecting the two equilibria.

    #     In the case of singletons equilibria, one can campute the integral of a function defined over the invariant set, called the "stability pressure":
    #     it will be positive/negative if the singleton is an unstable/stable boundary equilibria, and zero if the invariant branch does not intersect the safe set boundary; in this case, the removable area is also zero.

    #     This method implements a constrained optimization problem searching to find an invex CLF of the form V = √ m(x)' P m(x), P = N' G N
    #     that makes both the (i) removable areas and (ii) stability pressure non-negative for all branchs of the CLF-CBF pair branches, thus compatibilizing the CLF-CBF families.

    #     PS: the algorithm decision variable is N, a n x p matrix defining the CLF geometry through P = N' G N, where G > 0 is a n x n parameter matrix.
    #     '''
    #     np.set_printoptions(precision=4, suppress=True)

    #     G = self.comp_process_params["G"]
    #     Pinit = Ninit.T @ G @ Ninit

    #     Dinit = np.array(self.sos_factorized(Ninit))
    #     D_eig = np.linalg.eigvals( Dinit)
    #     D_eig_bounds = [ min(D_eig), max(D_eig) ]

    #     print(f"Dinit bounds = {D_eig_bounds}")

    #     # Initialize CLF geometry
    #     self.P = Pinit
    #     self.update_invariant_set()
    #     is_original_compatible = self.is_compatible()

    #     self.counter = 0

    #     def objective(var: np.ndarray) -> float:
    #         ''' Minimizes the changes to the CLF geometry '''
    #         self.counter += 1

    #         N = self.var_to_N(var)
    #         P = N.T @ G @ N

    #         self.cost = np.linalg.norm(N - Ninit) + np.linalg.norm( np.linalg.eigvals(P) )
    #         return self.cost

    #     def invexity_constr(var: np.ndarray) -> float:
    #         ''' Keeps the CLF invex '''
    #         N = self.var_to_N(var)
    #         D = np.array(self.sos_factorized(N))

    #         self.invexity = np.linalg.eigvals(D)

    #         if np.abs(D_eig_bounds[0]) < np.abs(D_eig_bounds[1]):
    #             return min(np.linalg.eigvals( D - self.invex_tol ))
    #         else:
    #             return -max(np.linalg.eigvals( D + self.invex_tol ))

    #     def center_constr(var: np.ndarray):
    #         ''' Keeps the CLF centered '''
    #         N = self.var_to_N(var)
    #         m_center = self.kernel.function(center)
    #         P = N.T @ G @ N
    #         self.centering = m_center.T @ P @ m_center
    #         return self.centering

    #     def lambda_max_constr(var: np.ndarray) -> list[float]:
    #         ''' Avoids eigenvalues of P from exploding '''

    #         N = self.var_to_N(var)
    #         max_eig = self.comp_process_params["max_P_eigenvalue"]
    #         P = N.T @ G @ N

    #         self.max_eig_constr = max_eig - max(np.linalg.eigvals(P))
    #         return self.max_eig_constr

    #     def compatibilization_constr(var: np.ndarray) -> list[float]:
    #         ''' Forces stability pressure + removable measures to be positive in every branch, for every CBF '''

    #         N = self.var_to_N(var)
    #         self.P = N.T @ G @ N

    #         self.update_invariant_set()      # The invariant set must be updated to get the current state of the optimization

    #         self.stability_pressures, self.measures = np.zeros(self.num_cbfs), np.zeros(self.num_cbfs)
    #         for cbf_index in range(self.num_cbfs):
    #             seg_measures = []
    #             for seg in self.invariant_segs[cbf_index]:
    #                 self.stability_pressures[cbf_index] += seg["stability_pressure"]
    #                 seg_measures.append( seg["removable_measure"] )
    #             self.measures[cbf_index] = min(seg_measures)

    #         constraints = np.hstack([self.stability_pressures, self.measures ])
    #         print(constraints)
    #         return constraints

    #     if animate:
    #         self.comp_graphics["fig"], ax = plt.subplots(nrows=1, ncols=1)

    #         ax.set_title("Showing compatibilization process...")
    #         ax.set_aspect('equal', adjustable='box')
    #         ax.set_xlim(self.limits[0], self.limits[1])
    #         ax.set_ylim(self.limits[2], self.limits[3])
    #         self.init_comp_plot(ax)
    #         plt.pause(self.comp_process_data["gui_eventloop_time"])

    #     def intermediate_callback(res: np.ndarray):
    #         '''
    #         Callback for visualization of intermediate results (verbose or by animation).
    #         '''
    #         # print(f"Status = {status}")
    #         self.comp_process_data["execution_time"] += time.perf_counter() - self.comp_process_data["start_time"]
    #         self.comp_process_data["step"] += 1

    #         # for later json serialization...
    #         for cbf_index in range(self.num_cbfs):
    #             self.comp_process_data["invariant_segs_log"][cbf_index].append( [ seg.tolist() for seg in self.invariant_lines[cbf_index] ] )

    #         if verbose:
    #             print( f"Steps = {self.counter}" )
    #             print( f"λ(P) = {np.linalg.eigvals(self.P)}" )
    #             print( f"Cost = {self.cost}" )
    #             print( f"Invexity = {self.invexity}" )
    #             print( f"Centering = {self.centering}" )
    #             print( f"Stability pressures = {self.stability_pressures}")
    #             print( f"Removable measures = {self.measures}")
    #             print(self.comp_process_data["execution_time"], "seconds have passed...")

    #         if animate:
    #             self.update_comp_plot(ax)
    #             plt.pause(self.comp_process_data["gui_eventloop_time"])

    #         self.counter = 0
    #         self.comp_process_data["start_time"] = time.perf_counter()

    #     constraints = []
    #     constraints += [ {"type": "ineq", "fun": invexity_constr} ]
    #     constraints += [ {"type": "eq", "fun": center_constr} ]
    #     constraints += [ {"type": "ineq", "fun": lambda_max_constr} ]
    #     constraints += [ {"type": "ineq", "fun": compatibilization_constr} ]

    #     #--------------------------- Main compatibilization process ---------------------------
    #     print("Starting compatibilization process. This may take a while...")
    #     is_processed_compatible = self.is_compatible()
    #     self.comp_process_data["start_time"] = time.perf_counter()

    #     init_var = self.N_to_var(Ninit)
    #     sol = minimize( objective, init_var, constraints=constraints, callback=intermediate_callback, options={"ftol": 1e-4} )

    #     N = self.var_to_N( sol.x )
    #     self.P = N.T @ G @ N
    #     self.update_invariant_set()

    #     is_processed_compatible = self.is_compatible()
    #     # --------------------------- Main compatibilization process ---------------------------
    #     print(f"Compatibilization terminated with message: {sol.message}")

    #     message = "Compatibilization "
    #     if is_processed_compatible: message += "was successful. "
    #     else: message += "failed. "
    #     message += "Process took " + str(self.comp_process_data["execution_time"]) + " seconds."
    #     print(message)

    #     if animate: plt.pause(2)

    #     comp_result = {
    #                     # "opt_message": sol.message,
    #                     "kernel_dimension": self.kernel._num_monomials,
    #                     "P_original": Pinit.tolist(),
    #                     "P_processed": self.P.tolist(),
    #                     "is_original_compatible": is_original_compatible,
    #                     "is_processed_compatible": is_processed_compatible,
    #                     "execution_time": self.comp_process_data["execution_time"],
    #                     "num_steps": self.comp_process_data["step"],
    #                     "invariant_set_log": self.comp_process_data["invariant_segs_log"]
    #                 }

    #     return comp_result

    # def plot_removable_areas(self, ax, cbf_index):
    #     '''
    #     Plots the removable areas associated to the invariant sets
    #     '''
    #     rem_area_contour = ctp.contour_generator( x=self.xg, y=self.yg, z=self.area_function[cbf_index] )  # creates new contour_generator object for the area function
    #     area_boundary_filled = rem_area_contour.filled(0.0, np.inf)

    #     renderer = Renderer(figsize=(4, 2.5))
    #     renderer.filled(area_boundary_filled, rem_area_contour.fill_type, color="gold")
    #     renderer.show()

    # def init_comp_plot(self, ax):
    #     ''' Initialize compatibilization animation plot '''

    #     self.comp_graphics["text"] = ax.text(0.01, 0.99, str("Optimization step = 0"), ha='left', va='top', transform=ax.transAxes, fontsize=10)
    #     for cbf in self.cbfs:
    #         cbf.plot_levels(levels = [ -0.1*k for k in range(4,-1,-1) ], ax=ax, limits=self.limits)
    #     self.update_comp_plot(ax)

    # def update_comp_plot(self, ax):
    #     ''' Update compatibilization animation plot '''

    #     step = self.comp_process_data["step"]
    #     self.comp_graphics["text"].set_text(f"Optimization step = {step}")

    #     for cbf_index in range(self.num_cbfs):
    #         self.plot_invariant(ax, cbf_index)

    #     self.plot_attr(ax, "stable_equilibria", mcolors.BASE_COLORS["r"], 1.0)
    #     self.plot_attr(ax, "unstable_equilibria", mcolors.BASE_COLORS["g"], 0.8)

    #     for coll in self.comp_graphics["clf_artists"]:
    #         coll.remove()

    #     num_eqs = len(self.boundary_equilibria)
    #     if num_eqs:
    #         self.clf.set_params(P=self.P)
    #         self.clf.generate_contour()
    #         level = self.clf.function( self.boundary_equilibria[np.random.randint(0,num_eqs)]["x"] )
    #         self.comp_graphics["clf_artists"] = self.clf.plot_levels(levels = [ level ], ax=ax, limits=self.limits)

class InvexProgram():
    '''
    Class for testing the new invexification method.
    '''
    def __init__(self, kernel, **kwargs):

        if not isinstance( kernel, Kernel ):
            raise Exception("First argument must be a valid Kernel.")
        self.kernel = kernel

        self._init_optimization()

        ''' Default parameters '''
        self.opmode = 'costinvex'                               # options = { 'cost', 'invex' }
        self.fit_to = None
        self.points_to_fit = []
        '''
        self.points_to_fit should be a list of dicts of the form:
        { "point": [...], "level": lvl, "gradient": [...], "curvature": curv }
        '''
        for sdp_param in ('slack_gain', 'invex_gain', 'cost_gain'):
            self.sdp_params_cvxpy[sdp_param].value = 1.0

        self.center = np.zeros(self.n)
        self.m_center.value = self.kernel.function( self.center )
        self.Tol.value = np.zeros((self.q, self.q))
        self.vecTol.value = np.zeros(self.q)
        self.invex_mode = 'matrix'                          # options = { 'matrix', 'eigen' }
        self.clock = perf_counter()

        ''' Customize parameters '''
        self.set_param(**kwargs)

    def _init_optimization(self):

        self.n = self.kernel._dim
        self.p = self.kernel._num_monomials
        self.q = self.kernel.dim_det_kernel

        ''' Define integrator dynamics '''
        self.geom_shape = (self.p, self.p)
        self.state_shape = (self.n, self.p)
        self.invex_shape = (self.q, self.q)
        self.dynamics_dim = self.n * self.p
        init_state = np.zeros(self.dynamics_dim)
        init_ctrl = np.zeros(self.dynamics_dim)

        from dynamic_systems import Integrator
        self.geo_dynamics = Integrator( init_state, init_ctrl )

        ''' Define cvxpy variables/parameters '''

        # ---------------------------- Overall parameters ---------------------------
        self.sdp_params_cvxpy = { "slack_gain": cp.Parameter(nonneg=True),
                                  "invex_gain": cp.Parameter(nonneg=True),
                                  "cost_gain":  cp.Parameter(nonneg=True) }

        # ---------------------------- Decision variables ---------------------------
        self.U = cp.Variable( self.state_shape )
        self.slack = cp.Variable(1)
        self.P = cp.Variable( (self.p, self.p), PSD=True )

        '''
        Define invexity constraint
        (either a Matrix Barrier Function or regular barrier with the eigenvalues)
        -> This constraint is the STAR of the party!
        If it works, it will allow for a search in the space of invex functions.
        '''
        invex_gain = self.sdp_params_cvxpy["invex_gain"]

        # ----------------------- For Matrix Barrier Function ------------------------
        self.cone = cp.Parameter()                                          # +1/-1 for PSD/NSD cones, respectively
        self.D = cp.Parameter( self.invex_shape )
        self.Tol = cp.Parameter( self.invex_shape, PSD=True )
        self.D_diff = [ [ cp.Parameter( self.invex_shape ) for _ in range(self.p) ] for _ in range(self.n) ]

        Ddot = 0
        for i, gradD_line in enumerate(self.D_diff):
            for j, gradD in enumerate(gradD_line):
                Ddot += gradD * self.U[i][j]

        M = self.cone * self.D - self.Tol
        Mdot = self.cone * Ddot
        self.matrix_invex_constraint = Mdot + invex_gain * M >> 0

        # ----------------------- For Vector Barrier Function ------------------------
        self.vecTol = cp.Parameter( self.q )
        self.lambdaD = cp.Parameter( self.q )
        self.lambdaD_diff = [ [ cp.Parameter( self.q ) for _ in range(self.p) ] for _ in range(self.n) ]

        lambdaDdot = 0
        for i, gradlambdaD_line in enumerate(self.lambdaD_diff):
            for j, gradlambdaD in enumerate(gradlambdaD_line):
                lambdaDdot += gradlambdaD * self.U[i][j]

        vecBarrier = self.cone * self.lambdaD - self.vecTol
        vecBarrierDot = self.cone * lambdaDdot
        self.vector_invex_constraint = vecBarrierDot + invex_gain * vecBarrier >= 0

        # --------------------------- Fixing the center ------------------------------
        self.m_center = cp.Parameter( self.p )
        self.center_constraint = self.U @ self.m_center == 0

        '''
        Define scalar constraint for minimization of total cost
        (Control Lyapunov Function). This constraint allows for
        a search in the space of geometries that minimize the sums of psd costs.
        '''
        self.tcost = cp.Parameter()
        self.tcost_diff = cp.Parameter(self.state_shape)

        tcost_dot = 0
        for (i,j) in itertools.product( range(self.n), range(self.p) ):
            tcost_dot += self.tcost_diff[i][j] * self.U[i][j]

        tcost_gain = self.sdp_params_cvxpy["cost_gain"]
        # self.cost_constraint = tcost_dot + tcost_gain * self.tcost <= 0.0
        self.cost_constraint = tcost_dot + tcost_gain * self.tcost <= self.slack

        # ----------------------- Define the SDP cost (minimizing the energy of geometry dynamics) ---------------------------
        slack_gain = self.sdp_params_cvxpy["slack_gain"]
        self.sdp_cost = cp.norm(self.U,'fro') + slack_gain * (self.slack)**2

        self.best_result = { "cost": np.inf, "invex_gap": np.inf, "control_energy": np.inf, "N": np.zeros(self.state_shape) }

    def _init_geometry(self):
        ''' Computes the initial invex geometry '''

        tol = 2e-1

        pts = [ pt['point'] for pt in self.points_to_fit ]
        H, center = stationary_volume_ellipsoid( pts, mode='min' )
        eigH, eigvecH = np.linalg.eig( tol*H )
        Pnom = kernel_quadratic(eigen=eigH, R=eigvecH.T, center=center, kernel_dim=self.p)
        self.Nnom, lowrank_error = NN_decomposition(Pnom, self.n)
        self.N = self.Nnom

        eigP = np.real(np.linalg.eigvals(self.N.T @ self.N))
        index, = np.where( eigP > 1e-3 )
        print(f"Initial λ(N'N) = {eigP[index]}")

        ''' Sets integrator state '''
        self.geo_dynamics.set_state( self.vec( self.N ) )

    def set_param(self, **kwargs):

        for key in kwargs.keys():
            key = key.lower()
            if key == 'fit_to':
                self.fit_to = kwargs[key]
                continue
            if key in ('slack_gain', 'invex_gain', 'cost_gain'):
                self.sdp_params_cvxpy[key].value = kwargs[key]
                continue
            if key == 'invex_tol':
                invex_tol = kwargs[key]
                self.Tol.value = np.zeros((self.q, self.q))
                self.Tol.value[0,0] = invex_tol
                self.vecTol.value = invex_tol*np.ones(self.q)
                continue
            if key == 'center':
                self.center = kwargs[key]
                self.m_center.value = self.kernel.function(self.center)
                self._init_geometry()
                continue
            if key == 'points':
                self.points_to_fit = kwargs[key]
                continue
            if key == 'mode':
                self.opmode = ''
                for t in kwargs[key]:
                    self.opmode += t       # ('cost','invex','center')
                continue
            if key == 'invex_mode':
                self.invex_mode = kwargs[key]
                continue

        ''' Sets up the cvxpy problem accordingly '''

        self.basic_constraints = []
        self.basic_constraints.append(self.center_constraint)

        # if 'invex' in self.opmode:
        #     if self.invex_mode == 'matrix':
        #         print("MBF-based invexification is ON.")
        #         self.basic_constraints.append( self.matrix_invex_constraint )
        #     if self.invex_mode == 'eigen':
        #         print("Eigen-based invexification is ON.")
        #         self.basic_constraints.append( self.vector_invex_constraint )
        # else:
        #     print("Invexification is OFF.")

        # if 'cost' in self.opmode:
        #     self.basic_constraints.append(self.cost_constraint)
        #     n_pts = len(self.points_to_fit)
        #     print(f"Cost minimization is ON, fitting {n_pts} points.")
        # else:
        #     print("Cost minimization is OFF.")

        self.problem = cp.Problem( cp.Minimize( self.sdp_cost ), constraints=self.basic_constraints )

    def is_invex(self, verbose=False):
        ''' Checks if current geometry is invex '''

        eigD = np.linalg.eigvals( self.kernel.D( self.N ) )
        psd, nsd = np.all( eigD >= 0.0 ), np.all( eigD <= 0.0 )
        invex = psd or nsd

        if verbose:
            if invex: print("Geometry is invex.")
            else: print("Geometry is not invex.")

        return invex

    def mat(self, state):
        state = np.array(state)
        if len(state) != self.dynamics_dim:
            raise Exception("State vector has incorrect dimensions.")
        return state.reshape(self.state_shape)

    def vec(self, N):
        N = np.array(N)
        if N.shape != self.state_shape:
            raise Exception("N has incorrect dimensions.")
        return N.flatten()

    def update_geom(self):

        ''' Updates D(N) and ∇D(N) (for MBF-based invexity) '''
        if self.invex_mode == 'matrix':

            self.D.value, D_diff = self.invex_barrier()
            for i,j in itertools.product( range(self.n), range(self.p) ):
                self.D_diff[i][j].value = D_diff[i][j]

        ''' Updates λ(D)(N) and ∇λ(D)(N) (for eigen-based invexity) '''
        if self.invex_mode == 'eigen':

            self.lambdaD.value, lambdaD_diff = self.invex_barrier()
            for i,j in itertools.product( range(self.n), range(self.p) ):
                self.lambdaD_diff[i][j].value = lambdaD_diff[i][j]

        '''
        Updates c(N) and ∇c(N) (for fitting many points)
        For now, only level fitting is fully implemented.
        '''
        self.tcost.value, self.tcost_diff.value = self.fitting_cost()

    def invex_barrier(self):
        '''
        Returns: (i)  D(self.N), ∇D(self.N)_ij (matrices), if option == 'matrix'
                 (ii) λ(D)(self.N), ∇λ(self.N)_ij (vectors), if option == 'eigen'
                 (iii) minλ(D)(self.N), corresponding ∇λ(self.N)_ij (scalar), if option == 'scalar'
        '''
        D = self.kernel.D(self.N)
        D_diff = self.kernel.D_diff(self.N)

        if self.invex_mode == 'matrix':
            return D, D_diff

        if self.invex_mode == 'eigen':
            eigvals, eigvecs = np.linalg.eig(D)
            lambdaD = np.zeros(self.q)
            lambdaD_diff = [ [ np.zeros(self.q) for _ in range(self.p) ] for _ in range(self.n) ]
            for k, eigD in enumerate(eigvals):
                v = eigvecs[:,k]
                lambdaD[k] = eigD
                for i,j in itertools.product( range(self.n), range(self.p) ):
                    lambdaD_diff[i][j][k] = v.T @ D_diff[i][j] @ v
            return lambdaD, lambdaD_diff

        ''' STILL NOT FULLY IMPLEMENTED '''
        if self.invex_mode == 'scalar':
            eigvals, eigvecs = np.linalg.eig(D)
            index = np.argmin(eigvals)
            lambdaD = eigvals[index]
            lambdaD_diff = [ [ 0.0 for _ in range(self.p) ] for _ in range(self.n) ]
            v = eigvecs[:,index]
            for i,j in itertools.product( range(self.n), range(self.p) ):
                lambdaD_diff[i,j] = v.T @ D_diff[i][j] @ v

            return lambdaD, lambdaD_diff

    def fitting_cost(self):
        '''
        Total fitting cost: c(N) = Σ_i ci(N)
        '''
        cost, cost_diff = 0.0, np.zeros(self.state_shape)
        # cost, cost_diff = self.trace_cost()
        # cost, cost_diff = self.comparison_cost()

        for point in self.points_to_fit:

            keys = point.keys()

            if "point" not in keys:
                raise Exception("Point coordinates must be specified.")

            pt = point["point"]

            if "level" in keys:
                level = point["level"]
                pt_cost, pt_cost_diff = self.level_cost(pt, level)
                cost += pt_cost
                cost_diff += pt_cost_diff

            if "gradient" in keys:
                direction = point["gradient"]
                if "norm" in keys:
                    norm = point["norm"]
                    pt_cost, pt_cost_diff = self.gradient_cost(pt, direction, norm)
                else:
                    pt_cost, pt_cost_diff = self.gradient_cost(pt, direction)
                cost += pt_cost
                cost_diff += pt_cost_diff

            if "curvature" in keys:
                direction = point["gradient"]
                curvature = point["curvature"]
                pt_cost, pt_cost_diff = self.curvature_cost(pt, direction, curvature)
                cost += pt_cost
                cost_diff += pt_cost_diff

        N = len(self.points_to_fit)
        cost *= 1/N
        cost_diff *= 1/N

        return cost, cost_diff

    def trace_cost(self):
        '''
        Cost tr(P) cost.
        Based on the heuristic that minimizing tr(P) will often lead to a low rank solution for P.
        '''
        P = self.N.T @ self.N
        cost = np.trace(P)
        cost_diff = np.zeros(self.state_shape)
        EYEn = np.eye(self.n)
        EYEp = np.eye(self.p)
        for (i,j) in itertools.product( range(self.n), range(self.p) ):
            eij = np.outer( EYEn[:,i], EYEp[:,j] )
            cost_diff[i,j] = np.trace( self.N.T @ eij + eij.T @ self.N )

        return cost, cost_diff

    def comparison_cost(self):
        '''
        Cost for minimizing ||N - Nnom||²
        '''
        cost = 0.5*np.linalg.norm(self.N - self.Nnom, 'fro')
        cost_diff = np.zeros(self.state_shape)
        for (i,j) in itertools.product( range(self.n), range(self.p) ):
            gradNij = np.zeros(self.state_shape)
            gradNij[i,j] = 1.0
            cost_diff[i,j] = self.vec(self.N - self.Nnom).T @ self.vec(gradNij)

        return cost, cost_diff

    def level_cost(self, point, level, mode='normal'):
        '''
        Cost for fitting a point to a particular level set.
        c(N) = 0.5 ( m(x)' N'N m(x) - lvl )²
        '''

        if self.fit_to == None: const = level
        if self.fit_to == 'clf': const = level**2
        if self.fit_to == 'cbf': const = 2*level + 1

        m = self.kernel.function(point)

        if mode == 'normal':
            error = 0.5 * ( m.T @ self.N.T @ self.N @ m - const )
            cost = 0.5 * error**2
            cost_diff = np.zeros(self.state_shape)
            for (i,j) in itertools.product( range(self.n), range(self.p) ):
                Nm = self.N @ m
                cost_diff[i,j] = error * Nm[i] * m[j]
            return cost, cost_diff

        if mode == 'cvxpy':
            cost = cp.norm(  m.T @ self.P @ m - const )
            return cost

    def gradient_cost(self, point, direction, grad_norm=1.0):
        '''
        Cost for fitting the gradient norm on a point to a particular value.
        c(N) = || ∇f ||² - norm²
        TO BE IMPLEMENTED
        '''
        m = self.kernel.function(point)
        Jm = self.kernel.jacobian(point)

        gradient = Jm.T @ self.N.T @ self.N @ m
        normalized = direction/np.linalg.norm(direction)

        grad_error = gradient - grad_norm * normalized

        cost = 0.5 * grad_error.T @ grad_error
        cost_diff = np.zeros(self.state_shape)

        return cost, cost_diff

    def curvature_cost(self, point, gradient, curv):
        '''
        Cost for fitting the value of the directional curvature on a point along a
        particular perpendicular direction to the gradient.
        c(N) = ( m(x)' S(N,v) m(x) - curv )²    ( fit CLF/CBF curvature 'curv' at direction 'v' to point x )
        TO BE IMPLEMENTED
        '''
        return 0.0, np.zeros(self.state_shape)

    def run_dynamic_opt(self, Ninit=None, verbose=False):

        self.invex_mode = 'matrix'

        if Ninit is not None: self.N = Ninit

        step_size = 1e-2
        step = 0
        self.running = True
        while self.running:

            step += 1

            ''' Computes everything with current N '''
            self.update_geom()

            ''' Find U control (solve SDP with current N) '''
            try:
                self.problem.solve(verbose=False, solver=cp.CLARABEL)
            except cp.SolverError as error:
                print(error)

            ''' Main integration step (update N state with computed U control) '''
            Uflatten = self.U.value.flatten()
            self.geo_dynamics.set_control( Uflatten )

            self.geo_dynamics.actuate( dt=step_size )
            self.N = self.mat( self.geo_dynamics.get_state() )

            ''' If is optimal, ends optimization '''
            if self.is_optimal(verbose=verbose):
                self.running = False

        return self.best_result["N"]

    def run_standard_opt(self, Ninit=None, verbose=False):
        ''' Run standard optimization '''

        self.invex_mode = 'eigen'

        if Ninit is not None: self.N = Ninit

        def cost(var):
            self.N = self.mat(var)

            cost, cost_diff = self.fitting_cost()
            return (cost, cost_diff)

        def invexity_barrier(var):
            self.N = self.mat(var)

            lambdaD, lambdaD_diff = self.invex_barrier()
            barrier = self.cone.value * lambdaD - self.vecTol.value
            return barrier

        def invexity_jac(var):
            self.N = self.mat(var)

            lambdaD, lambdaD_diff = self.invex_barrier()
            lambdaD_jac = np.rollaxis( np.array(lambdaD_diff), 2, 0 ).reshape( self.q, self.dynamics_dim )
            barrier_jac = self.cone.value * lambdaD_jac

            return barrier_jac

        '''
        This method computes the matrix A(xc) for the linear constraint
        A(xc) vec(N) == 0, equivalent to N m(xc) = 0 (the center constraint)'''
        def get_A(center):
            mc = np.array(self.kernel.function(center))
            A = sp.linalg.block_diag(*[ mc for _ in range(self.n) ])
            return A

        def show_message(var):

            N = self.mat(var)
            D = self.kernel.D(N)
            eigD = np.linalg.eigvals(D)

            if verbose:
                if platform.system().lower() != 'windows':
                    os.system('var=$(tput lines) && line=$((var-2)) && tput cup $line 0 && tput ed')           # clears just the last line of the terminal
                cost_val, cost_diff_val = cost(var)
                message = f"Total cost = {cost_val:.6f}, "
                message += f"λ(D)(N) = {np.sort(eigD)}, "
                message += f"||N|| = {np.linalg.norm(N):.3f}, "
                print(message)

        center_constr = LinearConstraint(get_A(self.center), lb=0.0, ub=0.0)
        invexity_constr = NonlinearConstraint(invexity_barrier, lb=0.0, ub=np.inf, jac=invexity_jac)
        constrs = [ center_constr, invexity_constr ]

        init_var = self.vec(self.N)
        sol = minimize( cost, init_var, constraints=constrs, method='SLSQP', jac=True, callback=show_message,
                        options={"disp": True, 'ftol': 1e-6, 'maxiter': 1000} )
        N = self.mat( sol.x )
        show_message(sol.x)

        return N

    def run_bilinear_opt(self, Ninit=None, verbose=False, jac=False):
        ''' Run bilinear optimization '''

        if Ninit is not None: self.Nnom = Ninit

        mcenter = np.array(self.kernel.function(self.center))
        A_list = self.kernel.get_A_matrices()
        r = self.kernel._jacobian_dim

        def R_blocks(N):
            P = N.T @ N
            R_blocks = [[ Ai.T @ P @ Aj for Aj in A_list ] for Ai in A_list ]
            return R_blocks

        def Rfun(N):
            R_blks = R_blocks(N)
            R = [[ None for _ in range(self.n) ] for _ in range(self.n) ]
            for i in range(self.n):
                for j in range(self.n):
                    R_blk = R_blks[i][j]
                    R[i][j] = R_blk[0:r,0:r]
                    R[i][j][0,0] -= mcenter.T @ R_blk @ mcenter

            return np.block(R)

        def eigRfun(var):
            self.N = self.mat(var)

            R = Rfun(self.N)
            eigR = np.sort( np.linalg.eigvals( R ).real )

            return eigR

        def cost(var):
            ''' Computes the fitting cost '''
            self.N = self.mat(var)
            cost, cost_diff = self.fitting_cost()
            if jac: return (cost, cost_diff)
            return cost

        max_lambda = 10.0
        def lambda_max(var):
            ''' Computes the lambda max constraint '''
            self.N = self.mat(var)
            return np.max( self.N.T @ self.N )

        def invexity_barrier(var):
            ''' Computes the invexity bilinear inequality constraint '''
            return np.min( eigRfun(var) )

        def get_center_matrix():
            '''
            Computes the matrix A(xc) for the linear constraint A(xc) vec(N) == 0,
            equivalent to N m(xc) = 0 (the center constraint).
            '''
            A = sp.linalg.block_diag(*[ mcenter for _ in range(self.n) ])
            return A

        def show_message(var, verbose=False, initial_message=''):
            if not verbose: return

            if platform.system().lower() != 'windows':
                os.system('var=$(tput lines) && line=$((var-2)) && tput cup $line 0 && tput ed')           # clears just the last line of the terminal

            if jac: cost_val, _ = cost(var)
            else: cost_val = cost(var)
            eigRvals = eigRfun(var)
            message = initial_message
            message += f" cost = {cost_val}\n"
            message += f"R(N) eigenvalues = {np.around(eigRvals,5)}\n"
            print(message)

        center_constr = LinearConstraint(get_center_matrix(), lb=0.0, ub=0.0)
        invexity_constr = NonlinearConstraint(invexity_barrier, lb=0.0, ub=np.inf)

        constrs = [ center_constr, invexity_constr ]

        init_var = self.vec(self.N)
        show_message(init_var, verbose=True, initial_message='Initial')

        sol = minimize( cost, init_var, constraints=constrs, method='SLSQP', jac=jac,
                        callback=lambda var: show_message(var, verbose),
                        options={"disp": True, 'maxiter': 400, 'ftol': 1e-12} )
        self.N = self.mat( sol.x )
        show_message( sol.x, verbose=True, initial_message='Final' )

        # P = self.N.T @ self.N
        # eigsP = np.linalg.eigvals(P).real
        # print(f"Eigs of P = {np.sort(eigsP)}")
        # eigsNN = np.linalg.eigvals(self.Nnom.T @ self.Nnom).real
        # print(f"Eigs of N'N = {np.sort(eigsNN)}")

        return self.N

    def run_sdp_opt(self, Pinit=None, verbose=False):

        center = self.kernel.function(self.center)
        mcenter = np.array(center)
        A_list = self.kernel.get_A_matrices()
        r = self.kernel._jacobian_dim

        # self.P.value = self.Nnom.T @ self.Nnom

        ''' CVXPY function for cost '''
        def cost_fun():

            cost = 0.0
            for point in self.points_to_fit:

                keys = point.keys()
                if "point" not in keys:
                    raise Exception("Point coordinates must be specified.")
                pt = point["point"]

                if "level" in keys:
                    level = point["level"]
                    cost += self.level_cost(pt, level, mode='cvxpy')

            return cost

        def R_blocks():
            R_blocks = [[ Ai.T @ self.P @ Aj for Aj in A_list ] for Ai in A_list ]

            return R_blocks

        def Rfun():
            S = np.vstack([ np.eye(r), np.zeros((self.p-r,r))])
            R_blks = R_blocks()
            R = [[ None for _ in range(self.n) ] for _ in range(self.n) ]
            for i in range(self.n):
                for j in range(self.n):
                    R_blk = R_blks[i][j]
                    vec = cp.hstack([ mcenter.T @ R_blk @ mcenter ] + [ 0.0 for _ in range(r-1) ])
                    R[i][j] = S.T @ R_blk @ S - cp.diag(vec)
            return cp.bmat(R)

        constraints = []

        ''' Upperbound constraint '''
        pts = [ pt["point"] for pt in self.points_to_fit ]
        H, center = stationary_volume_ellipsoid( pts, mode='max' )
        eigH, eigvecH = np.linalg.eig( H )
        Pnom = kernel_quadratic(eigen=eigH, R=eigvecH.T, center=center, kernel_dim=self.p)
        Nnom, lowrank_error = NN_decomposition(Pnom, self.n)

        ''' Invexity constraints '''
        # if np.all(Pinit != None):
        #     self.P.value = Pinit

        constraints += [ Rfun() >> 0 ]
        constraints += [ self.P @ mcenter == 0 ]
        constraints += [ self.P << Nnom.T @ Nnom ]
        # constraints += [ cp.lambda_max(self.P) <= 1.0 ]

        cost = cost_fun()
        prob = cp.Problem( cp.Minimize(cost), constraints=constraints )
        try:
            prob.solve(verbose=verbose, solver=cp.MOSEK)
        except cp.SolverError as error:
            print(error)

        P = self.P.value
        eigsP = np.linalg.eigvals( P )
        eigsNN = np.linalg.eigvals( Nnom.T @ Nnom )

        self.N, lowrank_error = NN_decomposition(P, self.n)
        print(f"Cost = {cost.value}")
        print(f"N'N = {np.sort(eigsNN)}")
        print(f"P = {np.sort(eigsP)}")

        return self.N

    def solve_program(self):
        '''
        BEST RESULT SO FAR!!! Forcing ∇Φ(x)' ∇Φ(x) - ∇Φ(x_c)' ∇Φ(x_c) >> 0 for all x !!!
        '''
        try:
            self.run_sdp_opt( verbose=True )
            # self.run_bilinear_opt( verbose=False )
        except KeyboardInterrupt:
            pass

        return self.N.T @ self.N

    def is_optimal(self, verbose=False):
        ''' Checks if optimization has reached its optimal value '''

        Dvalue = self.kernel.D(self.N)

        curr_cost = self.tcost.value
        c = self.cone.value
        curr_invex_gap = np.linalg.norm( PSD_closest(c*Dvalue) - c*Dvalue, 'fro')
        curr_ctrl_energy = np.linalg.norm( self.U.value )

        cost_decreased = False
        if curr_cost < self.best_result["cost"]:
            self.best_result["cost"] = curr_cost
            cost_decreased = True
            self.clock = perf_counter()

        invex_gap_decreased = False
        if curr_invex_gap < self.best_result["invex_gap"]:
            self.best_result["invex_gap"] = curr_invex_gap
            invex_gap_decreased = True

        if curr_ctrl_energy < self.best_result["control_energy"]:
            self.best_result["control_energy"] = curr_ctrl_energy

        if all((cost_decreased, invex_gap_decreased)):
            self.best_result["N"] = self.N

        dt = perf_counter() - self.clock

        best_cost = self.best_result["cost"]
        best_invex_gap = self.best_result["invex_gap"]
        min_control_energy = self.best_result["control_energy"]

        if verbose:
            if platform.system().lower() != 'windows':
                os.system('var=$(tput lines) && line=$((var-2)) && tput cup $line 0 && tput ed')           # clears just the last line of the terminal
            message = f"Min. control energy = {min_control_energy:.5f}, "
            message += f"best cost = {best_cost:.8f}, "
            message += f"best λ(D)(N) gap = {best_invex_gap:.10f}, "
            message += f"||N|| = {np.linalg.norm(self.N):.3f}, "
            message += f"Δt = {dt}"
            print(message)

        stop_criteria = []
        stop_criteria += [ best_cost <= 1e-3 ]              # Stops if cost is smaller than threshold
        stop_criteria += [ best_invex_gap <= 1e-7 ]         # Stops if invex gap is smaller than threshold
        # stop_criteria += [ dt >= 10.0 ]                     # Stops if more than 5s have passed without any improvement

        return all(stop_criteria)

    def save(self, file_name=''):
        ''' Method for saving the computed shape into a file '''

        pass

    def standard_cost(self, fun, collection: list[dict]):
        '''
        Standard scalar psd cost (to be minimized), given by the general expression

        c(N) = 0.5 * Σ_i ei(N)' @ Mi @ ei(N) >= 0 ,
        ei(N) = F(N) @ vi - Mi^-1 @ wi      , with the following parameters:

        - vi     -> vector / scalar parameter with dimension vi.dim / -
        - wi     -> vector / scalar parameter with dimension wi.dim / -
        - Mi > 0 -> symmetric pd matrix / scalar parameter with dimension (wi.dim, wi.dim) / -
        - F(N)   -> matrix / vector function of N with dimension (wi.dim, vi.dim) / vi.dim

        The ei(N) are vector / scalar error functions, since the minimum of c(N) occurs at ei(N) = 0, i = 0, 1, ...

        This expression allows for many cost expressions,
        ranging from Frobenius norms of matrices to gradient collinearity costs.

        Three distinct types are possible, depending on the shape of F(N):
            i)  contravariant vector - F(N) is column vector: vi is a scalar, Mi is matrix /
            ii) covariant vector     - F(N) is row vector
            iii) tensor              - F(N) is a matrix
        The correct dimensions for each case are as indicated above.

        Parameters: - fun -> method implementing f(N) and ∇f(N).
                    - collection -> list of dicts {"M": Mi, "v": vi, "w": wi } containing the parameters {Mi, vi, wi} for each term of the sum.

        Returns: - c(N)  -> scalar cost function computed at self.N
                 - ∇c(N) -> array containing the cost function gradients wrt to Nij at self.N, with dimensions (self.n, self.p).
        '''

        F, F_diff = fun(self.N)

        cost = 0.0
        cost_diff = np.zeros((self.n, self.p))

        for item in collection:

            Mk, vk, wk = item["M"],item["v"], item["w"]

            ''' Compute cost and gradient '''
            sqMk = sp.linalg.sqrtm(Mk)
            if np.linalg.norm(wk) >= 0.0:
                error_k = sqMk @ ( F @ vk - np.linalg.inv(Mk) @ wk )
            else:
                error_k = sqMk @ F @ vk

            cost += 0.5 * float( ( error_k.T @ Mk @ error_k ).reshape(1,) )

            for (i,j) in itertools.product( range(self.n), range(self.p) ):
                cost_diff[i,j] += ( error_k.T @ Mk @ F_diff[i,j] ) @ vk

        return cost, cost_diff

        '''
        F(N) and ∇F(N) matrix / vector valued function used in standard cost expression.
        The following types of cost functions are supported
        and have suitable parametrizations using the standard_cost.

        1) c(N) = || F(N) - F₀ ||²                 ( minimize Frobenius norm distance from matrix F(N) to 'F₀' )
        2) c(N) = ( m(x)' N'N m(x) - lvl )²        ( fit CLF/CBF level set 'lvl' to point x )
        3) c(N) = || ∇f ||² - ( ∇f' n )² ,
              f = m(x)' N'N m(x)                   ( fit CLF/CBF gradient direction 'n' to point x )
        4) c(N) = ( m(x)' S(N,v) m(x) - curv )²    ( fit CLF/CBF curvature 'curv' at direction 'v' to point x )

        ... among others.
        '''

class CLBF(KernelQuadratic):
    '''
    Class for kernel-based Control Lyapunov Barrier Functions.
    '''
    def __init__(self, *args):
        super().__init__(*args)
        pass