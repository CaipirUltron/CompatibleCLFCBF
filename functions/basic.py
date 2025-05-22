import logging
import itertools
import numpy as np
import contourpy as ctp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from abc import ABC, abstractmethod
from dataclasses import dataclass

from common import *

def commutation_matrix(n):
    '''
    Generate commutation matrix K relating the vectorization of a matrix n x n matrix A with the vectorization of its transpose A', as
    vec(A') = K vec(A).
    '''
    # determine permutation applied by K
    w = np.arange(n * n).reshape((n, n), order="F").T.ravel(order="F")
    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(n * n)[w, :]

def vec(A):
    '''
    Vectorize matrix in a column-major form (Fortran-style).
    '''
    return A.flatten('F')

def mat(vec):
    '''
    De-vectorize a square matrix which was previously vectorized in a column-major form.
    '''
    n = np.sqrt(len(vec))
    if (not n.is_integer()):
        raise Exception('Input vector does not represent a vectorized square matrix.')
    n = int(n)
    return vec.reshape(n,n).T

@dataclass
class LeadingShape:
    ''' Data class for a leading shape (to be used as an approximation tool) '''
    shape: np.ndarray
    bound: str = ''
    approximate: bool = False

class Function(ABC):
    ''' 
    Implementation of abstract class for scalar functions of any input dimension.

    '''
    def __init__(self, **kwargs):

        # Initialize basic parameters
        self._dim = 2
        self._output_dim = 1
        self.color = mcolors.BASE_COLORS["k"]
        self.linestyle = "solid"
        self.alpha = 1.0
        self.limits = (-1,1,-1,1)
        self.spacing = 0.1

        self.set_params(**kwargs)

        if self._output_dim == 1: 
            self.generate_contour()

    def _validate(self, point):
        ''' Validates input data '''

        if self._dim != 1:
            if not isinstance(point, (list, tuple, np.ndarray)): 
                raise Exception("Input data point is not a numeric array.")
        else:
            if isinstance(point, (np.int64, np.float64, float, int)):
                point = np.array([point])

        if isinstance(point, (list, tuple)): 
            point = np.array(point)
        return point

    @abstractmethod
    def _function(self, point: np.ndarray) -> np.ndarray:
        '''
        Abstract implementation of function value. 
        Must receive point as input and return the corresponding function value.
        Overwrite on children classes.
        '''
        pass

    @abstractmethod
    def _gradient(self, point: np.ndarray) -> np.ndarray:
        '''
        Abstract implementation of gradient vector.
        Must receive point as input and return the corresponding gradient value.
        Overwrite on children classes.
        '''
        pass

    @abstractmethod
    def _jacobian(self, point: np.ndarray) -> np.ndarray:
        '''
        Abstract implementation of the Jacobian matrix. 
        Must receive point as input and return the corresponding gradient value.
        Overwrite on children classes.
        '''
        pass

    @abstractmethod
    def _hessian(self, point: np.ndarray) -> np.ndarray:
        '''
        Abstract implementation of hessian computation. Must receive point as input and return the corresponding hessian value.
        Overwrite on children classes.
        '''
        pass

    def generate_contour(self):
        '''
        Create contour generator object for the given function.
        Parameters: limits (2x2 array) - min/max limits for x,y coords
                    spacing - grid spacing for contour generation
        '''        
        if self._dim != 2:
            logging.warning("Contour plot can only be used for 2D functions.")
            self.contour = None
            return

        x_min, x_max, y_min, y_max = self.limits
        x = np.arange(x_min, x_max, self.spacing)
        y = np.arange(y_min, y_max, self.spacing)
        xg, yg = np.meshgrid(x,y)
        
        fvalues = np.zeros(xg.shape)
        for i,j in itertools.product(range(xg.shape[0]), range(xg.shape[1])):
            pt = np.array([xg[i,j], yg[i,j]])
            fvalues[i,j] = self(pt)
        
        self.contour = ctp.contour_generator(x=xg, y=yg, z=fvalues )

    def __call__(self, point):
        return self._function(self._validate(point))

    def function(self, point):
        return self._function(self._validate(point))

    def gradient(self, point):
        return self._gradient(self._validate(point))    

    def jacobian(self, point):
        return self._jacobian(self._validate(point))  

    def hessian(self, point):
        return self._hessian(self._validate(point))

    def set_params(self, **params):
        ''' Sets function basic parameters (mostly plotting) '''

        for key in params.keys():
            key = key.lower()
            if key == "dim":
                self._dim = params["dim"]
                continue
            if key == "color":
                self.color = params["color"]
                continue
            if key == "linestyle":
                self.linestyle = params["linestyle"]
                continue
            if key == "limits":
                self.limits = params["limits"]
                continue
            if key == "spacing":
                self.spacing = params["spacing"]
                continue

    def get_levels(self, levels=[0.0] ) -> list:
        ''' Generates function level sets from the contour generator object '''
        if not self.contour: return []

        level_contours = []
        for lvl in levels:
            line = self.contour.lines(lvl)
            level_contours.append(line)
        return level_contours

    def plot_levels(self, ax = plt, levels=[0.0], **kwargs):
        ''' Plots function level sets at the input axis ax. Additional args may be passed for color and linestyle '''
        
        color = self.color
        linestyle = self.linestyle
        alpha = self.alpha
        for key in kwargs.keys():
            key = key.lower()
            if key == "color":
                color = kwargs["color"]
                continue
            if key == "linestyle":
                linestyle = kwargs["linestyle"]
                continue
            if key == "alpha":
                alpha = kwargs["alpha"]
                continue

        collections = []
        for level in self.get_levels(levels):
            for segment in level:
                line2D = ax.plot( segment[:,0], segment[:,1], color=color, linestyle=linestyle, alpha=alpha )
                collections.append(line2D[0])
        return collections

class Quadratic(Function):
    '''
    Class for quadratic function representing x'Ax + b'x + c = 0.5 (x - p)'H(x-p) + height = 0.5 x'Hx - 0.5 p'(H + H')x + 0.5 p'Hp + height
    '''
    def to_factored(A,b,c):
        H = 2*A
        x0 = - np.linalg.inv(H) @ b
        min = c - 0.5 * x0.T @ H @ x0
        return H, x0, min
    
    def from_factored(H,x0,min):
        A = 0.5*H
        b = - H @ x0
        c = 0.5 * x0.T @ H @ x0 + min
        return A, b, c

    def __init__(self, **kwargs):
        ''' Initialize it with hessian, center and height '''
        super().__init__(**kwargs)

    def set_params(self, **kwargs):
        '''
        Sets the quadratic function parameters.
        '''
        super().set_params(**kwargs)
        for key in kwargs:
            if key == "hessian":
                self.H = np.array(kwargs[key])
                self._dim = self.H.shape[0]
                continue
            if key == "center":
                self.center = np.array(kwargs[key])
                self._dim = len(self.center)
                continue
            if key == "height":
                self.height = kwargs[key]
                continue

        self.A, self.b, self.c = Quadratic.from_factored( self.H, self.center, self.height )

    def _function(self, pt):
        '''
        General quadratic function.
        '''
        return np.array(pt) @ self.A @ np.array(pt) + self.b @ np.array(pt) + self.c

    def _gradient(self, pt):
        '''
        Gradient of general quadratic function.
        '''
        return ( self.A + self.A.T ) @ np.array(pt) + self.b

    def _jacobian(self, pt):
        return self._gradient(pt)

    def _hessian(self, pt):
        '''
        Hessian of general quadratic function.
        '''
        return ( self.A + self.A.T )

    def get_values(self, x):
        '''  
        Returns function, gradient and Hessian values at an input point x
        '''
        fun = self(x)
        nabla_fun = self.gradient(x)
        Hfun = self.hessian(x)
        return fun, nabla_fun, Hfun

    def eig(self):
        eigs, Q = np.linalg.eig(self.H)
        return eigs, Q

    def eigvals(self):
        eigs, Q = self.eig()
        return eigs

    def __str__(self):
        return f"Quadratic 0.5 (x - p)'H(x-p) with H = \n{self.H}\n and center = {self.center} "

class Gaussian(Function):
    '''
    Class for gaussian function of the type N(x) = c exp( -0.5 (x-mu).T Sigma (x-mu) )
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self.c = 0.0
        if self._dim > 1:
            self.mu = np.zeros(self._dim)
            self.Sigma = np.zeros([self._dim,self._dim])
        else:
            self.mu = 0.0
            self.Sigma = 0.0
        self.set_param(**kwargs)

        self.epsilon = 0.0
        self.Lv = sym2triangular( self._hessian-self.epsilon*np.eye(self._dim) )
        self.param = triangular2vector( self.Lv )
        # self.dynamics = Integrator(self.param,np.zeros(len(self.param)))

    def set_param(self, **kwargs):
        '''
        Sets the quadratic function parameters.
        '''
        for key in kwargs:
            if key == "constant":
                self.c = kwargs[key]
            if key == "mean":
                self.mu = np.array(kwargs[key])
            if key == "shape":
                if np.shape(kwargs[key]) != ( self._dim, self._dim ):
                    raise Exception('Shape matrix must be the same dimension as the mean.')
                self.Sigma = np.array(kwargs[key])

    def update(self, param_ctrl, dt):
        pass

    def function(self, point):
        '''
        Gaussian function.
        '''
        v = np.array(point) - self.mu
        return self.c * np.exp( -0.5 * v.T @ self.Sigma @ v )

    def gradient(self, point):
        '''
        Gradient of Gaussian function.
        '''
        v = np.array(point) - self.mu
        return - self.c * np.exp( -0.5 * v.T @ self.Sigma @ v ) * ( self.Sigma @ v )

    def hessian(self, point):
        '''
        Hessian of Gaussian function.
        '''
        v = np.array(point) - self.mu
        return - self.c * np.exp( -v.T @ self.Sigma @ v ) * ( self.Sigma - np.outer( self.Sigma @ v, self.Sigma @ v ) )

##################### OLD CODE #############################################################################################################

# class Function():
#     '''
#     Implementation of general class for scalar functions.
#     '''
#     def __init__(self, *args, **kwargs):

#         self.set_value(*args)
#         self.functions = []
#         self.functions.append(self)

#         self._function = 0.0
#         if self._dim > 1:
#             self._gradient = np.zeros(self._dim)
#             self._hessian = np.zeros([self._dim,self._dim])
#         else:
#             self._gradient = 0.0
#             self._hessian = 0.0

#         limits = None
#         spacing = 0.1
#         self.plot_config = {"color": mcolors.BASE_COLORS["k"], "linestyle": 'solid'}
#         for key in kwargs.keys():
#             if key == "limits":
#                 limits = kwargs["limits"]
#                 continue
#             if key == "spacing":
#                 spacing = kwargs["spacing"]
#                 continue
#             if key == "plot_config":
#                 self.plot_config = kwargs["plot_config"]
#                 continue

#         if limits != None:
#             self.gen_contour(limits, spacing=spacing)

#     def set_value(self, *args):
#         '''
#         Initialize values and returns corresponding points.
#         '''
#         data_type = np.array(args).dtype
#         if data_type != np.dtype('float64') and data_type != np.dtype('int64'):
#             raise Exception("Data type not understood.")

#         self._args = args
#         self._dim = len(args)
#         self._var = np.array(self._args).reshape(self._dim,-1)
#         self._num_points = np.size(self._var, 1)

#         return self.get_value()

#     def get_value(self):
#         return self._var

#     def evaluate(self):
#         self.evaluate_function(*self._var)
#         self.evaluate_gradient(*self._var)
#         self.evaluate_hessian(*self._var)

#     def evaluate_function(self, *args):
#         self.set_value(*args)
#         self.function_values()
#         return self.get_function()

#     def evaluate_gradient(self, *args):
#         self.set_value(*args)
#         self.gradient_values()
#         return self.get_gradient()

#     def evaluate_hessian(self, *args):
#         self.set_value(*args)
#         self.hessian_values()
#         return self.get_hessian()

#     def get_function(self):
#         '''
#         Get last computed function
#         '''
#         return self._function

#     def get_gradient(self):
#         '''
#         Get last computed gradient
#         '''
#         return self._gradient

#     def get_hessian(self):
#         '''
#         Get last computed hessian
#         '''
#         return self._hessian

#     def function_values(self):
#         '''
#         Compute function values.
#         '''
#         self._function = np.zeros(self._num_points)
#         for k in range(self._num_points):
#             fun_val = 0.0
#             for func in self.functions:
#                 fun_val += func.function(self._var[:,k])
#             self._function[k] = fun_val

#         return self._function

#     def gradient_values(self):
#         '''
#         Compute gradient values.
#         '''
#         self._gradient = []
#         for point in self._var.T:
#             grad = np.zeros(self._dim)
#             for func in self.functions:
#                 grad += func.gradient(point)
#             self._gradient.append(grad)

#         return self._gradient

#     def hessian_values(self):
#         '''
#         Compute hessian values.
#         '''
#         self._hessian = []
#         for point in self._var.T:
#             hess = np.zeros([self._dim,self._dim])
#             for func in self.functions:
#                 hess += func.gradient(point)
#             self._hessian.append(hess)

#         return self._hessian

#     def function(self, point):
#         '''
#         Abstract implementation of function computation. Must receive point as input and return the corresponding function value.
#         Overwrite on children classes.
#         '''
#         return 0.0

#     def gradient(self, point):
#         '''
#         Abstract implementation of gradient computation. Must receive point as input and return the corresponding gradient value.
#         Overwrite on children classes.
#         '''
#         return np.zeros(self._dim)

#     def hessian(self, point):
#         '''
#         Abstract implementation of hessian computation. Must receive point as input and return the corresponding hessian value.
#         Overwrite on children classes.
#         '''
#         return np.zeros([self._dim, self._dim])

#     def __add__(self, func):
#         '''
#         Add method.
#         '''
#         if not isinstance(func, Function):
#             raise Exception("Only Function objects can be summed.")

#         from copy import copy
#         function = copy(self)
#         function.functions.append(func)

#         return function

#     def gen_contour(self, limits, spacing=0.1):
#         '''
#         Create contour generator object for the given function.
#         Parameters: limits (2x2 array) - min/max limits for x,y coords
#                     spacing - grid spacing for contour generation
#         '''        
#         if self._dim != 2:
#             raise Exception("Contour plot can only be used for 2D functions.")

#         x_min, x_max = limits[0][0], limits[0][1]
#         y_min, y_max = limits[1][0], limits[1][1]

#         x = np.arange(x_min, x_max, spacing)
#         y = np.arange(y_min, y_max, spacing)
#         xg, yg = np.meshgrid(x,y)

#         mesh_fvalues = np.zeros([np.size(xg,0),np.size(xg,1)])
#         for i in range(np.size(xg,1)):
#             args = []
#             args.append(xg[:,i])
#             args.append(yg[:,i])
#             for k in range(self._dim-2):
#                 args.append( [self._var[k+2,0] for _ in range(len(xg[:,i]))] )
#             # mesh_fvalues[:,i] = np.array(self.evaluate_function(xv[:,i], yv[:,i]))
#             mesh_fvalues[:,i] = np.array(self.evaluate_function(*args))
        
#         self.contour = ctp.contour_generator(x=xg, y=yg, z=mesh_fvalues )
#         return self.contour

#     def get_levels(self, levels, **kwargs):
#         '''
#         Generates function level sets.
#         Parameters: levels (list of floats)
#         Returns: a list with all level segments, in the same order as levels
#         '''
#         limits = None
#         spacing = 0.1
#         for key in kwargs.keys():
#             aux_key = key.lower()
#             if aux_key == "limits":     # Must always be required if self.contours still does not exist
#                 limits = kwargs[key]
#                 continue
#             if aux_key == "spacing":    # Must always be required if self.contours still does not exist
#                 spacing = kwargs[key]
#                 continue

#         if not isinstance(limits, list):
#             if not hasattr(self, "contour"):
#                 raise Exception("Grid limits are required to create contours.")
#         else:
#             self.gen_contour(limits, spacing=spacing)

#         level_contours = []
#         for lvl in levels:
#             level_contours.append( self.contour.lines(lvl) )

#         return level_contours

#     def plot_levels(self, levels, **kwargs):
#         '''
#         Plots function level sets.
#         Parameters: levels (list of floats)
#         Returns: plot collections
#         '''
#         ax = plt
#         color = self.plot_config["color"]
#         linestyle = self.plot_config["linestyle"]
#         for key in kwargs.keys():
#             aux_key = key.lower()
#             if aux_key == "ax":
#                 ax = kwargs["ax"]
#                 continue
#             if aux_key == "color":
#                 color = kwargs["color"]
#                 continue
#             if aux_key == "linestyle":
#                 linestyle = kwargs["linestyle"]
#                 continue

#         collections = []
#         level_contours = self.get_levels(levels, **kwargs)
#         for level in level_contours:
#             for segment in level:
#                 line2D = ax.plot( segment[:,0], segment[:,1], color=color, linestyle=linestyle )
#                 collections.append(line2D[0])

#         return collections


# old methods from KernelTriplet
# class OLDKernelTripletMethops(KernelTriplet):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def update_segment(self, segment):
#         '''
#         Updates segment. Aims to significantly improve performance on updating the invariant set. 
#         '''
#         n = self.plant.n
#         seg_points = segment["points"]

#         num_pts = len(seg_points)
#         line_sizes = [ np.linalg.norm(seg_points[k] - seg_points[k+1]) for k in range(len(seg_points)-1) ]

#         def get_deltas(var):
#             return var[0:n*num_pts].reshape(num_pts, n)

#         def objective(var):
#             '''
#             var is a list with coordinates [ pt1[0] pt1[1] .. pt1[n-1] pt2[0] pt2[1] ... pt2[n-1] ...  ptm[n-1] l1 l2 ... lm ]
#             n is the state dimension
#             m is the number of points in the segment
#             '''
#             deltas = get_deltas(var)
#             fun = sum( map(lambda d: np.linalg.norm(d)**2, deltas) )

#             new_seg_points = seg_points + deltas
#             fun += sum( [ ( np.linalg.norm(new_seg_points[k] - new_seg_points[k+1]) - line_sizes[k] )**2 for k in range(num_pts-1) ] )
#             fun += sum( [ self.det_invariant( *new_seg_points[k].tolist() )**2 for k in range(num_pts) ] )
#             return fun
        
#         # def invariant_constr(var):
#         #     deltas = get_deltas(var)
#         #     return [ self.det_invariant( *(seg_points[k] + deltas[k]).tolist() ) for k in range(num_pts) ]
    
#         def lambda_constr(var):
#             deltas = get_deltas(var)
#             return [ self.compute_lambda( seg_points[k] + deltas[k] ) for k in range(num_pts) ]

#         # constraints = [ {"type": "eq", "fun": invariant_constr} ]
#         constraints = []
#         constraints.append( {"type": "ineq", "fun": lambda_constr} )

#         init_var = [ 0.0 for _ in range(n*num_pts) ]
#         sol = minimize(objective, init_var, constraints=constraints, options={"disp":True})

#         new_seg_pts = seg_points + get_deltas(sol.x)

#         segment["points"] = new_seg_pts
#         segment["lambdas"] = [ self.compute_lambda(new_seg_pts[k]) for k in range(num_pts) ]
#         segment["boundary_equilibria"] = self.seg_boundary_equilibria(new_seg_pts)
#         segment["interior_equilibria"] = self.seg_interior_equilibria(new_seg_pts)
#         return segment

#     def update_invariant_set(self):
#         '''
#         Updates the invariant set.
#         '''
#         self.boundary_equilibria = []
#         self.interior_equilibria = []
#         for segment in self.invariant_segs:
#             self.update_segment( segment )
#             self.boundary_equilibria += segment["boundary_equilibria"]
#             self.interior_equilibria += segment["interior_equilibria"]

#     def equilibria_from_invariant(self, verbose=False):
#         '''
#         Computes all equilibrium points and local branch optimizers of the CLF-CBF pair, using the invariant set intersections with the CBF boundary.
#         '''
#         if len(self.invariant_segs) == 0:
#             self.invariant_set(extended=False)

#         # Finds intersections between boundary and invariant set segments (boundary equilibria)
#         self.boundary_equilibria = []
#         self.interior_equilibria = []
#         for boundary_seg in self.boundary_segs:
#             for invariant_seg in self.invariant_segs:
                
#                 boundary_curve = geometry.LineString(boundary_seg)
#                 invariant_seg_curve = geometry.LineString(invariant_seg)
#                 intersections = intersection( boundary_curve, invariant_seg_curve )

#                 new_candidates = []
#                 if not intersections.is_empty:
#                     if hasattr(intersections, "geoms"):
#                         for geo in intersections.geoms:
#                             x, y = geo.xy
#                             x, y = list(x), list(y)
#                             new_candidates += [ [x[k], y[k]] for k in range(len(x)) ]
#                     else:
#                         x, y = intersections.xy
#                         x, y = list(x), list(y)
#                         new_candidates += [ [x[k], y[k]] for k in range(len(x)) ]
                
#                 for pt in new_candidates:

#                     eq_sol = self.optimize_over("boundary", init_x=pt)
#                     if (eq_sol) and "equilibrium" in eq_sol.keys():
#                         add_to(eq_sol, self.boundary_equilibria)

#                     eq_sol = self.optimize_over("interior", init_x=pt)
#                     if (eq_sol) and "equilibrium" in eq_sol.keys():
#                         add_to(eq_sol, self.interior_equilibria)

#                     branch_minimizer = self.optimize_over("min_branch", init_x=pt)
#                     if branch_minimizer and "type" in branch_minimizer.keys():
#                         add_to(branch_minimizer, self.branch_minimizers)

#                     branch_maximizer = self.optimize_over("max_branch", init_x=pt)
#                     if branch_maximizer and "type" in branch_maximizer.keys():
#                         add_to(branch_maximizer, self.branch_maximizers)

#         # self.branch_optimizers(verbose)

#         if verbose:
#             show_message(self.boundary_equilibria, "boundary equilibrium points")
#             show_message(self.interior_equilibria, "interior equilibrium points")

#             show_message(self.branch_minimizers, "branch minimizers")
#             show_message(self.branch_maximizers, "branch maximizers")

    # def equilibria(self, verbose=False):
    #     '''
    #     Computes all equilibrium points and local branch optimizers of the CLF-CBF pair, using the invariant set rectangular limits as initializers for the optimization algorithm.
    #     This method does not require the update of the complete invariant set geometry, 
    #     and is capable of computing the equilibrium points and local branch optimizers faster than the previous method.
    #     '''
    #     # Get initializers from boundary lines
    #     self.branch_initializers = [] 
    #     for line in self.limit_lines:
    #         self.branch_initializers += self.get_zero_det(line["x"], line["y"])

    #     # Find boundary, interior equilibria and branch optimizers
    #     self.boundary_equilibria = []
    #     self.interior_equilibria = []
    #     self.branch_minimizers = []
    #     self.branch_maximizers = []
    #     for pt in self.branch_initializers:

    #         eq_sol = self.optimize_over("boundary", init_x=pt)
    #         if (eq_sol) and "equilibrium" in eq_sol.keys():
    #             add_to(eq_sol, self.boundary_equilibria)

    #         eq_sol = self.optimize_over("interior", init_x=pt)
    #         if (eq_sol) and "equilibrium" in eq_sol.keys():
    #             add_to(eq_sol, self.interior_equilibria)

    #         branch_minimizer = self.optimize_over("min_branch", init_x=pt)
    #         if branch_minimizer and "type" in branch_minimizer.keys():
    #             add_to(branch_minimizer, self.branch_minimizers)

    #         branch_maximizer = self.optimize_over("max_branch", init_x=pt)
    #         if branch_maximizer and "type" in branch_maximizer.keys():
    #             add_to(branch_maximizer, self.branch_maximizers)

    #     # self.branch_optimizers(verbose)

    #     if verbose:
    #         show_message(self.boundary_equilibria, "boundary equilibrium points")
    #         show_message(self.interior_equilibria, "interior equilibrium points")

    #         show_message(self.branch_minimizers, "branch minimizers")
    #         show_message(self.branch_maximizers, "branch maximizers")

#     def branch_optimizers(self, verbose=False):
#         '''
#         Compute the branch optimizers
#         '''
#         self.connections_to_min = { i:[] for i in range(0,len(self.boundary_equilibria)) }
#         self.connections_to_max = { i:[] for i in range(0,len(self.boundary_equilibria)) }
#         self.branch_minimizers = []
#         self.branch_maximizers = []

#         # Create adjacency list for connections btw eq points and optimizers
#         for num_eq in range(len(self.boundary_equilibria)):
#             eq_sol = self.boundary_equilibria[num_eq]

#             branch_minimizer = self.optimize_over("min_branch", init_x=eq_sol["x"])
#             if branch_minimizer and "type" in branch_minimizer.keys():
#                 add_to(branch_minimizer, self.branch_minimizers, self.connections_to_min[num_eq])

#             branch_maximizer = self.optimize_over("max_branch", init_x=eq_sol["x"])
#             if branch_maximizer and "type" in branch_maximizer.keys():
#                 add_to(branch_maximizer, self.branch_maximizers, self.connections_to_max[num_eq])

#         # Checks if there exist removable optimizers
#         self.check_removables()

#         if verbose:
#             show_message(self.boundary_equilibria, "boundary equilibrium points")
#             show_message(self.interior_equilibria, "interior equilibrium points")

#             show_message(self.branch_minimizers, "branch minimizers")
#             show_message(self.branch_maximizers, "branch maximizers")

#             print(f"Connections to minimizers = {self.connections_to_min}")
#             print(f"Connections to maximizers = {self.connections_to_max}")

#     def check_removables(self):
#         '''
#         Checks if equilibrium point with index eq_index is removable.
#         Returns the corresponding minimizer/maximizer that removes the equilibrium point.
#         '''
#         self.min_removers, self.max_removers = [], []

#         for eq_index in range(len(self.boundary_equilibria)):
#             for minimizer_index in self.connections_to_min[eq_index]:
#                 for j in self.connections_to_min.keys():
#                     if j == eq_index:       # ignore if self
#                         continue
#                     if minimizer_index in self.connections_to_min[j] and np.linalg.norm( self.branch_minimizers[minimizer_index]["gradh"] ) > 1e-3:
#                         self.branch_minimizers[minimizer_index]["type"] = "remover"
#                         add_to(self.branch_minimizers[minimizer_index], self.min_removers)
#                         break

#             for maximizer_index in self.connections_to_max[eq_index]:
#                 for j in self.connections_to_max.keys():
#                     if j == eq_index:       # ignore if self
#                         continue
#                     if maximizer_index in self.connections_to_max[j] and np.linalg.norm( self.branch_maximizers[maximizer_index]["gradh"] ) > 1e-3:
#                         self.branch_maximizers[maximizer_index]["type"] = "remover"
#                         add_to(self.branch_maximizers[maximizer_index], self.max_removers)
#                         break

#     def optimize_over(self, optimization=None, **kwargs):
#         '''
#         Finds equilibrium points solutions using sliding mode control. If no initial point is specified, it selections a point at random from a speficied interval.
#         Returns a dict containing all relevant data about the found equilibrium point, including its stability.
#         '''
#         init_x_def = False
#         for key in kwargs.keys():
#             aux_key = key.lower()
#             if aux_key == "init_x":
#                 init_x = kwargs[key]
#                 init_x_def = True
#                 continue

#         if not init_x_def:
#             init_x = [ np.random.uniform( self.limits[k][0], self.limits[k][1] ) for k in range(self.n) ]

#         def invariant_set(var):
#             '''
#             Returns the vector residues of invariant set -> is zero for x in the invariant set
#             '''
#             x = var[0:self.n]
#             return det_invariant(x, self.kernel, self.P, self.cbf.Q, self.plant.get_F(), self.params)

#         def boundary_constraint(var):
#             '''
#             Returns the diff between mQm and 1
#             '''
#             x = var[0:self.n]
#             delta = var[self.n]

#             h = self.cbf.function(x)
#             return delta - np.abs(h)

#         def objective(var):
#             '''
#             Objective function to be minimized
#             '''
#             delta = var[self.n]
#             x = var[0:self.n]
            
#             if optimization == "boundary":
#                 return delta**2
#             elif optimization == "interior":
#                 return self.compute_lambda(x.tolist())**2
#             elif optimization == "min_branch":
#                 return self.cbf.function(x)
#             elif optimization == "max_branch":
#                 return -self.cbf.function(x)
#             else: 1.0

#         init_delta = 1.0
#         init_var = init_x + [init_delta]

#         constraints = [ {"type": "eq", "fun": invariant_set} ]
#         if optimization == "boundary":
#             constraints.append({"type": "ineq", "fun": boundary_constraint})

#         sol = minimize(objective, init_var, constraints=constraints)

#         eq_coords = sol.x[0:self.n].tolist()
#         l = self.compute_lambda(eq_coords)
#         h = self.cbf.function(eq_coords)
#         gradh = self.cbf.gradient(eq_coords)

#         sol_dict = None

#         # Valid solution is a point in the invariant set with lambda >= 0
#         if l >= 0 and np.abs(invariant_set(sol.x)) < 1e-3:
#             sol_dict = {}
#             sol_dict["x"] = eq_coords
#             sol_dict["lambda"] = l
#             sol_dict["delta"] = sol.x[self.n]
#             sol_dict["invariant_cost"] = invariant_set(sol.x)
#             sol_dict["h"] = h
#             sol_dict["gradh"] = np.linalg.norm(gradh)
#             sol_dict["init_x"] = init_x
#             # sol_dict["message"] = sol.message
        
#         # Boundary equilibrium point - compute stability
#         if (sol_dict) and (np.abs(sol_dict["h"]) <= 1e-3):
#             stability, eta = self.compute_stability(eq_coords, "boundary")
#             sol_dict["eta"], sol_dict["stability"] = eta, stability
#             sol_dict["equilibrium"] = "stable"
#             if stability > 0:
#                 sol_dict["equilibrium"] = "unstable"

#         # Interior equilibrium points (for now, stability is not computed)
#         if (sol_dict) and (optimization == "interior") and (np.abs(sol_dict["lambda"]) <= 1e-5):
#             stability, eta = self.compute_stability(eq_coords, "interior")
#             sol_dict["eta"], sol_dict["stability"] = eta, stability
#             sol_dict["equilibrium"] = "stable"
#             if stability > 0:
#                 sol_dict["equilibrium"] = "unstable"

#         # Minimizers
#         if (sol_dict) and optimization == "min_branch":
#             if sol_dict["gradh"] < 1e-03:
#                 sol_dict["type"] = "cbf_minimum"
#             else: sol_dict["type"] = "undefined"

#         # Maximizers
#         if (sol_dict) and optimization == "max_branch":
#             if sol_dict["h"] > 1e+05 or sol_dict["gradh"] > 1e+06:
#                 sol_dict = None                 # filters unbounded maximizers
#             else: sol_dict["type"] = "undefined"

#         return sol_dict

    # def get_zero_det(self, xg, yg):
    #     '''
    #     Returns the points where the determinant is zero over a 1D array with coords given by xg, yg 
    #     '''
    #     det_grid = self.det_invariant(xg, yg, extended=True)
    #     indexes = np.where(np.sign(det_grid[:-1]) != np.sign(det_grid[1:]))[0] + 1

    #     pts = []
    #     for i in indexes:
    #         pts.append( [xg[i][0], yg[i][0]] )
    #     return pts
    
