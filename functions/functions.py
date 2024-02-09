import math
import itertools
import numpy as np
import scipy as sp
import cvxpy as cp
import sympy
import warnings

import contourpy as ctp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from common import *
from dynamic_systems import Integrator

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

class Function():
    '''
    Implementation of general class for scalar functions.
    '''
    def __init__(self, *args):

        self.set_value(*args)
        self.functions = []
        self.functions.append(self)

        self._function = 0.0
        if self._dim > 1:
            self._gradient = np.zeros(self._dim)
            self._hessian = np.zeros([self._dim,self._dim])
        else:
            self._gradient = 0.0
            self._hessian = 0.0

    def set_value(self, *args):
        '''
        Initialize values and returns corresponding points.
        '''
        data_type = np.array(args).dtype
        if data_type != np.dtype('float64') and data_type != np.dtype('int64'):
            raise Exception("Data type not understood.")

        self._args = args
        self._dim = len(args)
        self._var = np.array(self._args).reshape(self._dim,-1)
        self._num_points = np.size(self._var, 1)

        return self.get_value()

    def get_value(self):
        return self._var

    def evaluate(self):
        self.evaluate_function(*self._var)
        self.evaluate_gradient(*self._var)
        self.evaluate_hessian(*self._var)

    def evaluate_function(self, *args):
        self.set_value(*args)
        self.function_values()
        return self.get_function()

    def evaluate_gradient(self, *args):
        self.set_value(*args)
        self.gradient_values()
        return self.get_gradient()

    def evaluate_hessian(self, *args):
        self.set_value(*args)
        self.hessian_values()
        return self.get_hessian()

    def get_function(self):
        '''
        Get last computed function
        '''
        return self._function

    def get_gradient(self):
        '''
        Get last computed gradient
        '''
        return self._gradient

    def get_hessian(self):
        '''
        Get last computed hessian
        '''
        return self._hessian

    def function_values(self):
        '''
        Compute function values.
        '''
        self._function = np.zeros(self._num_points)
        for k in range(self._num_points):
            fun_val = 0.0
            for func in self.functions:
                fun_val += func.function(self._var[:,k])
            self._function[k] = fun_val

        return self._function

    def gradient_values(self):
        '''
        Compute gradient values.
        '''
        self._gradient = []
        for point in self._var.T:
            grad = np.zeros(self._dim)
            for func in self.functions:
                grad += func.gradient(point)
            self._gradient.append(grad)

        return self._gradient

    def hessian_values(self):
        '''
        Compute hessian values.
        '''
        self._hessian = []
        for point in self._var.T:
            hess = np.zeros([self._dim,self._dim])
            for func in self.functions:
                hess += func.gradient(point)
            self._hessian.append(hess)

        return self._hessian

    def function(self, point):
        '''
        Abstract implementation of function computation. Must receive point as input and return the corresponding function value.
        Overwrite on children classes.
        '''
        return 0.0

    def gradient(self, point):
        '''
        Abstract implementation of gradient computation. Must receive point as input and return the corresponding gradient value.
        Overwrite on children classes.
        '''
        return np.zeros(self._dim)

    def hessian(self, point):
        '''
        Abstract implementation of hessian computation. Must receive point as input and return the corresponding hessian value.
        Overwrite on children classes.
        '''
        return np.zeros([self._dim, self._dim])

    def __add__(self, func):
        '''
        Add method.
        '''
        if not isinstance(func, Function):
            raise Exception("Only Function objects can be summed.")

        from copy import copy
        function = copy(self)
        function.functions.append(func)

        return function

    def gen_contour(self, limits, spacing=0.1):
        '''
        Create contour generator object for the given function.
        '''
        if self._dim != 2:
            raise Exception("Contour plot can only be used for 2D functions.")
        n = 2

        x_min, x_max = limits[0][0], limits[0][1]
        y_min, y_max = limits[1][0], limits[1][1]

        x = np.arange(x_min, x_max, spacing)
        y = np.arange(y_min, y_max, spacing)
        xg, yg = np.meshgrid(x,y)

        mesh_fvalues = np.zeros([np.size(xg,0),np.size(xg,1)])
        for i in range(np.size(xg,1)):
            args = []
            args.append(xg[:,i])
            args.append(yg[:,i])
            for k in range(self._dim-2):
                args.append( [self._var[k+2,0] for _ in range(len(xg[:,i]))] )
            # mesh_fvalues[:,i] = np.array(self.evaluate_function(xv[:,i], yv[:,i]))
            mesh_fvalues[:,i] = np.array(self.evaluate_function(*args))
        
        self.contour = ctp.contour_generator(x=xg, y=yg, z=mesh_fvalues )
        # self.contour_segments = self.contour.lines(0.0)

    def plot_levels(self, levels, **kwargs):
        '''
        Method for plotting level sets.
        Returns: QuadContour object for 2D contour plots.
        '''
        if self._dim != 2:
            raise Exception("Contour plot can only be used for 2D functions.")
        n = 2

        resolution = 0.1
        colors = 'k'
        ax = plt
        for key in kwargs.keys():
            aux_key = key.lower()
            if aux_key == "ax":
                ax = kwargs[key]
                continue
            if aux_key == "resolution":
                resolution = kwargs[key]
                continue
            if aux_key == "colors":
                colors = kwargs[key]
                continue

        limits = [ [-1, +1] for _ in range(n) ]
        if "limits" in kwargs.keys():
            limits = kwargs["limits"]
            for i in range(n):
                if limits[i][0] >=  limits[i][1]:
                    raise Exception("Lines should be sorted in ascending order.")

        x_min, x_max = limits[0][0], limits[0][1]
        y_min, y_max = limits[1][0], limits[1][1]

        x = np.arange(x_min, x_max, resolution)
        y = np.arange(y_min, y_max, resolution)
        xv, yv = np.meshgrid(x,y)

        mesh_fvalues = np.zeros([np.size(xv,0),np.size(xv,1)])
        for i in range(np.size(xv,1)):
            args = []
            args.append(xv[:,i])
            args.append(yv[:,i])
            for k in range(self._dim-2):
                args.append( [self._var[k+2,0] for _ in range(len(xv[:,i]))] )
            # mesh_fvalues[:,i] = np.array(self.evaluate_function(xv[:,i], yv[:,i]))
            mesh_fvalues[:,i] = np.array(self.evaluate_function(*args))

        return ax.contour(xv, yv, mesh_fvalues, levels=levels, colors=colors)

class Quadratic(Function):
    '''
    Class for quadratic function representing x'Ax + b'x + c = 0.5 (x - p)'H(x-p) + height = 0.5 x'Hx - 0.5 p'(H + H')x + 0.5 p'Hp + height
    '''
    def __init__(self, *args):

        # Set parameters
        super().__init__(*args)

        if self._dim > 1:
            self.A = np.zeros([self._dim,self._dim])
            self.b = np.zeros(self._dim)
            self.critical_point = np.zeros(self._dim)
            self.dcritical = np.zeros(self._dim)
        else:
            self.A = 0.0
            self.b = 0.0
            self.critical_point = 0.0
            self.dcritical = 0.0
        self.c = 0.0
        self.height = 0.0

        # self.set_param(kwargs)

        # Set eigenbasis for hessian matrix
        _, _, Q = self.compute_eig()
        self.eigen_basis = np.zeros([self._dim, self._dim, self._dim])
        for k in range(self._dim):
            self.eigen_basis[:][:][k] = np.outer( Q[:,k], Q[:,k] )

    def set_param(self, **kwargs):
        '''
        Sets the quadratic function parameters.
        '''
        for key in kwargs:
            if key == "hessian":
                self._hessian = np.array(kwargs[key])
            if key == "critical":
                self.critical_point = np.array(kwargs[key])
            if key == "height":
                self.height = kwargs[key]
            if key == "dcritical":
                self.dcritical = np.array(kwargs[key])

        self.A = 0.5 * self._hessian
        self.b = - 0.5*( self._hessian + self._hessian.T ) @ self.critical_point
        self.c = 0.5 * self.critical_point @ ( self._hessian @ self.critical_point ) + self.height

        for key in kwargs:
            if key == "A":
                self.A = kwargs[key]
            if key == "b":
                self.b = kwargs[key]
            if key == "c":
                self.c = kwargs[key]

    def function(self, point):
        '''
        General quadratic function.
        '''
        return np.array(point) @ ( self.A @ np.array(point) ) + self.b @ np.array(point) + self.c

    def gradient(self, point):
        '''
        Gradient of general quadratic function.
        '''
        return ( self.A + self.A.T ) @ np.array(point) + self.b

    def hessian(self, point):
        '''
        Hessian of general quadratic function.
        '''
        return ( self.A + self.A.T )

    def eigen2hessian(self, eigen):
        '''
        Returns hessian matrix from a given set of eigenvalues.
        '''
        if self._dim != len(eigen):
            raise Exception("Dimension mismatch.")

        H = np.zeros(self._dim)
        for k in range(self._dim):
            H = H + eigen[k] * self.eigen_basis[:][:][k]

        return H

    def get_critical(self):
        return self.critical_point

    def get_height(self):
        return self.height

    def get_critical_derivative(self):
        return self.dcritical

    def compute_eig(self):
        eigen, Q = np.linalg.eig(self.hessian(0))
        angle = np.arctan2(Q[0, 1], Q[0, 0])
        return eigen, angle, Q

class QuadraticLyapunov(Quadratic):
    '''
    Class for Quadratic Lyapunov functions of the type (x-x0)'Hv(x-x0), parametrized by vector pi_v.
    Here, the Lyapunov minimum is a constant vector x0, and the hessian Hv is positive definite and parametrized by:
    Hv = Lv(pi_v)'Lv(pi_v) + epsilon I_n (Lv is upper triangular and epsilon is a small positive constant).
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        super().set_param(**kwargs)
        super().set_param(height=0.0)

        self.epsilon = 0.0
        self.Lv = sym2triangular( self._hessian-self.epsilon*np.eye(self._dim) )
        self.param = triangular2vector( self.Lv )
        self.dynamics = Integrator(self.param,np.zeros(len(self.param)))

    def get_param(self):
        '''
        This function gets the params corresponding to the Lyapunov Hessian matrix.
        '''
        return self.param

    def set_epsilon(self, epsilon):
        '''
        Sets the minimum eigenvalue for the Lyapunov Hessian matrix.
        '''
        self.epsilon = epsilon
        self.set_param(self.param)

    def set_param(self, param):
        '''
        Sets the Lyapunov function parameters.
        '''
        self.param = param
        Lv = vector2triangular(param)
        Hv = Lv.T @ Lv + self.epsilon*np.eye(self._dim)
        super().set_param(hessian = Hv)

    def set_critical(self, pt):
        '''
        Sets the Lyapunov function critical point.
        '''
        super().set_param(critical = pt)

    def set_critical_derivative(self, dcritical):
        '''
        Sets the derivative of the Lyapunov function critical point.
        '''
        super().set_param(dcritical = dcritical)

    def update(self, param_ctrl, dt):
        '''
        Integrates the parameters.
        '''
        self.dynamics.set_control(param_ctrl)
        self.dynamics.actuate(dt)
        self.set_param(self.dynamics.get_state())

    def get_partial_Hv(self):
        '''
        Returns the partial derivatives of Hv wrt to the parameters.
        '''
        tri_basis = triangular_basis(self._dim)
        partial_Hv = np.zeros([ len(self.param), self._dim, self._dim ])
        for i in range(len(self.param)):
            for j in range(len(self.param)):
                partial_Hv[i,:,:] = partial_Hv[i,:,:] + ( tri_basis[i].T @ tri_basis[j] + tri_basis[j].T @ tri_basis[i] )*self.param[j]

        return partial_Hv

class QuadraticBarrier(Quadratic):
    '''
    Class for Quadratic barrier functions. For positive definite Hessians, the unsafe set is described by the interior of an ellipsoid.
    The symmetric Hessian is parametrized by Hh(pi) = sum^n_i Li pi_i, where {Li} is the canonical basis of the space of (n,n) symmetric matrices.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        super().set_param(**kwargs)
        super().set_param(height = -0.5)

        self.param = sym2vector(self._hessian)
        self.dynamics = Integrator(self.param,np.zeros(len(self.param)))
        self.last_gamma_sol = 0.0
        self.gamma_min = -np.pi/2
        self.gamma_max = +np.pi/2

    def get_param(self):
        '''
        This function gets the params corresponding to the barrier Hessian matrix.
        '''
        return self.param

    def set_param(self, param):
        '''
        Sets the barrier function parameters.
        '''
        self.param = param
        super().set_param(hessian = vector2sym(param))

    def set_critical(self, pt):
        '''
        Sets the barrier function critical point.
        '''
        super().set_param(critical = pt)

    def update(self, param_ctrl, dt):
        '''
        Integrates the barrier function parameters.
        '''
        self.dynamics.set_control(param_ctrl)
        self.dynamics.actuate(dt)
        self.set_param(self.dynamics.get_state())

    def barrier_set(self, set_parameters):
        '''
        Computes the barrier with respect to a set define by set_parameters
        '''
        r = set_parameters["radius"]
        p_center = set_parameters["center"]
        theta = set_parameters["orientation"]

        def compute_pt(gamma):
            return np.array(p_center) + r*rot2D(theta) @ np.array([np.cos(gamma), np.sin(gamma)]).reshape(2)

        def necessary_cond(gamma):
            p_gamma = compute_pt( gamma )
            nablah = self.evaluate_gradient(*p_gamma)[0]
            return nablah.dot(np.array([-np.sin(theta+gamma), np.cos(theta+gamma)]))

        def cost(gamma):
            p_gamma = compute_pt(gamma)
            return self.evaluate_function(*p_gamma)[0]

        # Solve optimization problem
        import scipy.optimize as opt
        results = opt.minimize( cost, self.last_gamma_sol )
        gamma_opt = results.x[0]

        h = cost(gamma_opt)

        p_gamma_opt = compute_pt( gamma_opt )
        nablah = self.evaluate_gradient(*p_gamma_opt)[0]

        self.last_gamma_sol = gamma_opt

        return h, nablah, p_gamma_opt, gamma_opt

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
        self.dynamics = Integrator(self.param,np.zeros(len(self.param)))

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

class Kernel(Function):
    '''
    Class for kernel functions m(x) of maximum degree 2*d, where m(x) is a vector of (n+d,d) known monomials.
    '''
    def __init__(self, *args, **kwargs):

        # Initialization
        super().__init__(*args)
        self.set_param(**kwargs)

        # Create symbols
        import sympy as sp
        self._symbols = []
        for dim in range(self._dim):
            self._symbols.append( sp.Symbol('x' + str(dim+1)) )

        # Generate monomial list and symbolic monomials
        self.alpha, self.powers_by_degree = generate_monomial_list( self._dim, self._degree )
        self._monomials = generate_monomials_from_symbols( self._symbols, self.alpha )
        self._num_monomials = len(self._monomials)
        self._K = commutation_matrix(self._num_monomials)       # commutation matrix to be used later

        # Symbolic computations
        self._symP = sp.Matrix(sp.symarray('p',(self._num_monomials,self._num_monomials)))
        self._sym_monomials = sp.Matrix(self._monomials)
        self._sym_jacobian_monomials = self._sym_monomials.jacobian(self._symbols)

        self._hessian_monomials = [ [0 for i in range(self._dim)] for j in range(self._dim) ]
        for i in range(self._dim):
            for j in range(self._dim):
                self._hessian_monomials[i][j] = sp.diff(self._sym_jacobian_monomials[:,j], self._symbols[i])

        # Compute numeric A and N matrices
        self.compute_A()
        self.compute_N()

        # Lambda functions
        self._lambda_monomials = sp.lambdify( list(self._symbols), self._monomials )
        self._lambda_jacobian_monomials = sp.lambdify( list(self._symbols), self._sym_jacobian_monomials )
        self._lambda_hessian_monomials = sp.lambdify( list(self._symbols), self._hessian_monomials )

    def set_param(self, **kwargs):
        '''
        Sets the function parameters.
        '''
        self._degree = 0
        self._num_monomials = 1
        self._maxdegree = 2*self._degree

        for key in kwargs:
            if key == "degree":
                self._degree = kwargs[key]
                self._num_monomials = num_comb(self._dim, self._degree)
                self._coefficients = np.zeros([self._num_monomials, self._num_monomials])

        self.kernel_dim = self._num_monomials

    def compute_A(self):
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

    def compute_N(self):
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

    def function(self, point):
        '''
        Compute polynomial function numerically.
        '''
        m = np.array(self._lambda_monomials(*point))
        return m

    def jacobian(self, point):
        '''
        Compute kernel Jacobian.
        '''
        Jac_m = np.array(self._lambda_jacobian_monomials(*point))
        return Jac_m

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
        F, _ = kernel_constraints( point, self.powers_by_degree )
        return F

    def get_matrix_constraints(self):
        '''
        Returns kernel constraints
        '''
        from common import kernel_constraints
        _, matrices = kernel_constraints( np.zeros(self.kernel_dim), self.powers_by_degree )
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
        F, _ = kernel_constraints( point, self.powers_by_degree )
        if np.linalg.norm(F) < 0.00000001:
            return True
        else:
            return False

    def __eq__(self, other):
        '''
        Determines if two kernels are the same.
        '''
        return np.all( self.alpha == other.alpha )

    def __str__(self):
        '''
        Prints kernel
        '''
        variables = str(self._symbols)
        kernel = str(self._monomials)
        text = "m: R^" + str(self._dim) + " --> R^" + str(self._num_monomials) + "\nKernel map on variables " + variables + "\nm(x) = " + kernel
        return text

class KernelQuadratic(Function):
    '''
    Class for kernel quadratic functions of the type f(x) = m(x)' F m(x) - C for a given kernel m(x), where:
    F is a p.s.d. matrix and C is an arbitrary constant. If no constant C is specified, C = 0
    '''
    def __init__(self, *args, **kwargs):

        # Initialization
        super().__init__(*args)

        self.default_fit_options = {"force_coords": False, "force_gradients": False }
        self.fit_options = self.default_fit_options

        default_color = "black"
        if isinstance(self, KernelLyapunov):
            default_color = mcolors.TABLEAU_COLORS['tab:blue']
        elif isinstance(self, KernelBarrier):
            default_color = mcolors.TABLEAU_COLORS['tab:red']
        self.default_plot_config = {"figsize": (5,5), "axeslim": (-6,6,-6,6), "color": default_color}
        self.plot_config = self.default_plot_config

        self.set_param(**kwargs)
        self.evaluate()

        if len(self.points) > 0 or type(self.cost) != float or len(self.constraints) > 1:
            self.fit()

    def init_kernel(self):
        # very time the Kernel is initialized, self.matrix_coefs gets the correct dimensions and goes to zero.
        self.kernel_dim = self.kernel.kernel_dim
        self.matrix_coefs = np.zeros([self.kernel_dim, self.kernel_dim])

        self.param_dim = int(self.kernel_dim*(self.kernel_dim + 1)/2)
        self.dynamics = Integrator( np.zeros(self.param_dim), np.zeros(self.param_dim) )

        self.SHAPE = cp.Variable( (self.kernel_dim,self.kernel_dim), symmetric=True )
        self.clear_optimization()

    def clear_optimization(self):
        '''
        Clear optimization. Initially, cost is zero and the only constraint should be self.SHAPE >> 0
        '''
        self.points = []
        self.cost = 0.0
        self.constraints = [ self.SHAPE >> 0 ]

    def set_param(self, **kwargs):
        '''
        Sets the function parameters.
        '''
        if "constant" in kwargs.keys():
            self.constant = kwargs["constant"]

        if not hasattr(self, "constant"):
            self.constant = 0.0

        if "kernel" in kwargs.keys():
            if type(kwargs["kernel"]) != Kernel:
                raise Exception("Argument must be a valid Kernel function.")
            self.kernel = kwargs["kernel"]
            self.init_kernel()

        if "degree" in kwargs.keys():
            self.kernel = Kernel(*self._args, degree=kwargs["degree"])
            self.init_kernel()

        # Only initializes the standard kernel iff nothing was passed upon creation
        if not hasattr(self, "kernel"):
            self.kernel = Kernel(*self._args, degree=1)
            self.init_kernel()

        for key in kwargs.keys():

            if key in ["constant", "kernel", "degree"]: # Already dealt with
                continue

            if key == "fit_options":
                for default_key in self.default_fit_options.keys():
                    if default_key in kwargs["fit_options"].keys():
                        self.fit_options[default_key] = kwargs["fit_options"][default_key]
                continue

            if key == "plot_config":
                for default_key in self.default_plot_config.keys():
                    if default_key in kwargs["plot_config"].keys():
                        self.plot_config[default_key] = kwargs["plot_config"][default_key]
                continue

            if key == "coefficients":
                matrix_coefs = np.array(kwargs["coefficients"])

                if matrix_coefs.ndim != 2:
                    raise Exception("Matrix of coefficients must be a two-dimensional array.")
                if matrix_coefs.shape[0] != matrix_coefs.shape[1]:
                    raise Exception("Matrix of coefficients must be a square.")
                # if not np.all(np.linalg.eigvals(matrix_coefs) >= -1e-5):
                #     raise Exception("Matrix of coefficients must be positive semi-definite.")
                if not np.all( matrix_coefs == matrix_coefs.T ):
                    warnings.warn("Matrix of coefficients is not symmetric. The symmetric part will be used.")
                if matrix_coefs.shape[0] != self.kernel_dim:
                    raise Exception("Matrix of coefficients doesn't match the kernel dimension.")

                self.matrix_coefs = 0.5 * ( matrix_coefs + matrix_coefs.T )

                if isinstance(self, KernelLyapunov):
                    self.P = self.matrix_coefs
                if isinstance(self, KernelBarrier):
                    self.Q = self.matrix_coefs

                self.param = sym2vector( self.matrix_coefs )
                self.dynamics.set_state(self.param)
                continue

            if key == "points":
                self.points += kwargs["points"]
                continue

            if key == "centers":
                for center in kwargs["centers"]:
                    self.points.append({"coords": center, "level":-self.constant})
                continue

            if key == "leading":
                if "shape" not in kwargs["leading"].keys():
                    raise Exception("Must specify a shape matrix for the leading function.")
                leading_shape = kwargs["leading"]["shape"]

                if "uses" not in kwargs["leading"].keys():
                    raise Exception("Must specify a use for the leading function.")
                uses = kwargs["leading"]["uses"]
                
                for use in uses:
                    if use not in ["lower_bound", "upper_bound", "approximation"]:
                        raise Exception("Invalid use for the leading function.")
                
                bound = 0
                if "lower_bound" in uses: bound = -1
                if "upper_bound" in uses: bound = +1

                approx = False
                if "approximation" in uses or ( "lower_bound" in uses and "upper_bound" in uses ):
                    approx = True

                self.leading_function(leading_shape, bound=bound, approximate=approx)

        if self.matrix_coefs.shape != (self.kernel_dim, self.kernel_dim):
            raise Exception("P must be (p x p), where p is the kernel dimension!")

    def update(self, param_ctrl, dt):
        '''
        Integrates the parameters.
        '''
        self.dynamics.set_control(param_ctrl)
        self.dynamics.actuate(dt)
        new_param = self.dynamics.get_state()
        self.set_param( coefficients = vector2sym(new_param) )

    def is_sos_convex(self, verbose=False):
        '''
        Returns True if the function is SOS convex.
        '''
        sos_convex = False
        A_list = self.kernel.get_A_matrices()
        SOSConvexMatrix = np.block([[ Ai.T @ self.matrix_coefs @ Aj + Aj.T @ Ai.T @ self.matrix_coefs for Aj in A_list ] for Ai in A_list ])
        eigs = np.linalg.eigvals(SOSConvexMatrix)
        if np.all(eigs >= 0.0):
            sos_convex = True

        if verbose:
            if sos_convex: print(f"{self} is SOS convex.")
            else: print(f"{self} is not SOS convex, with negative eigenvalues = {eigs[eigs < 0.0]}")

        return sos_convex

    def fit(self):
        '''
        Fits the coefficient matrix to a list of desired points.
        Parameters: uses the list 
        points = [ { "point"     : ArrayLike, 
                     "level"     : float >= -self.constant, 
                     "gradient"  : ArrayLike, 
                     "curvature" : float }, ... ]
        to add more constraints or terms to the cost function before trying to solve the optimization. 
        Returns: the optimization results.
        '''
        n = self._dim
        p = self.kernel_dim
        A_list = self.kernel.get_A_matrices()

        # Iterate over the input list to get problem requirements
        gradient_norms = []
        for pt in self.points:
            keys = pt.keys()

            if "coords" not in keys:
                raise Exception("The point coordinates must be specified.")
            if "force_coord" not in keys:
                pt["force_coord"] = False
            if "force_gradient" not in keys:
                pt["force_gradient"] = False

            coords = pt["coords"]
            if type(coords) == np.ndarray:
                coords = coords.tolist()

            m = self.kernel.function(coords)
            Jm = self.kernel.jacobian(coords)

            # Define point-level constraints
            if "level" in keys:
                level_value = pt["level"]
                if level_value >= -self.constant:
                    if self.fit_options["force_coords"] or pt["force_coord"]:
                        self.constraints += [ 0.5 * m.T @ self.SHAPE @ m - self.constant == level_value ]
                    else:
                        self.cost += ( 0.5 * m.T @ self.SHAPE @ m - self.constant - level_value )**2
                else:
                    continue

            # Define gradient constraints
            if "gradient" in keys:
                gradient_norms.append( cp.Variable() )
                gradient = np.array(pt["gradient"])
                normalized = gradient/np.linalg.norm(gradient)

                if self.fit_options["force_gradients"] or pt["force_gradient"]:
                    self.constraints += [ Jm.T @ self.SHAPE @ m == gradient_norms[-1] * normalized ]
                else:
                    self.cost += cp.norm( Jm.T @ self.SHAPE @ m - gradient_norms[-1] * normalized )
                self.constraints += [ gradient_norms[-1] >= 0 ]

            # Define curvature constraints (2D only)
            if "curvature" in keys:
                if n != 2:
                    raise Exception("Error: curvature fitting was not implemented for dimensions > 2. ")
                if "gradient" not in keys:
                    raise Exception("Cannot specify a curvature without specifying the gradient.")

                curvature = pt["curvature"]
                v = rot2D(np.pi/2) @ normalized

                curvature_var = 0.0
                for i,j in itertools.product(range(n),range(n)):
                    Hij = m.T @ ( A_list[i].T @ self.SHAPE + self.SHAPE @ A_list[i] ) @ A_list[j] @ m
                    curvature_var += Hij * v[i] * v[j]
                self.cost += ( curvature_var - curvature )**2

        fit_problem = cp.Problem( cp.Minimize( self.cost ), self.constraints )
        fit_problem.solve()

        if "optimal" in fit_problem.status:
            if isinstance(self, KernelLyapunov):
                print("Lyapunov fitting was successful with final cost = " + str(fit_problem.value) + " and message: " + str(fit_problem.status))
            elif isinstance(self, KernelBarrier):
                print("Barrier fitting was successful with final cost = " + str(fit_problem.value) + " and message: " + str(fit_problem.status))
            else:
                print("Function fitting was successful with final cost = " + str(fit_problem.value) + " and message: " + str(fit_problem.status))
            self.set_param( coefficients = self.SHAPE.value )
            return fit_problem
        else:
            raise Exception("Problem is " + fit_problem.status + ".")
        
    def leading_function(self, Pleading, bound=0, approximate=False):
        '''
        Defines a leading function. Can be used as an lower bound, upper bound or as an approximation.
        Parameters: Pleading = (p x p) np.ndarray, where p is the kernel space dimension
                    bound = int (< 0, 0, >0): if zero, no bound occurs. If negative/positive, passed function is a lower/upper bound.
                    approximate = bool: if the function must be approximated.
        '''
        if bound != 0 or approximate:
            if Pleading.shape[0] != Pleading.shape[1]:
                raise Exception("Shape matrix for the bounding function must be square.")
            if Pleading.shape != (self.kernel_dim, self.kernel_dim):
                raise Exception("Shape matrix and kernel dimensions are incompatible.")

            if bound > 0:
                self.constraints += [ self.SHAPE >> Pleading ]
            elif bound < 0:
                self.constraints += [ self.SHAPE << Pleading ]

            if approximate:
                self.cost += cp.norm( self.SHAPE - Pleading )

    def define_level_set(self, points, level, contained=False):
        '''
        Adds points and constraints (if contained=True) with specific level set to self.points. 
        Flag contained=True ensures that the passed points are completely contained in the level set.
        Parameters: points -> list of points /
                              dict with { "coords": list (mandatory), 
                                          "gradient": list (optional),
                                          "curvature": float (optional) }
                    level  -> value of level set
        Returns: the optimization error.
        '''
        for pt in points:

            if isinstance(pt, list) or isinstance(pt, np.ndarray):
                self.points.append( {"coords": pt, "level": level} )
            elif isinstance(pt, dict):
                pt_dict = pt
                pt_dict["level"] = level
                self.points.append(pt_dict)
            else:
                raise Exception("Must pass list of points!")
            
            if contained:
                m = self.kernel.function(self.points[-1]["coords"])
                self.constraints.append( m.T @ self.SHAPE @ m <= 1.0 )

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

    def enclosing_quadratic(self):
        '''
        Find the P matrix of a quadratic enclosing all defined points in self.points
        '''
        num_pts = len(self.points)
        points = np.zeros([num_pts, self._dim])
        levels = []
        for k in range(num_pts):
            pt = self.points[k]
            points[k,:] = np.array(pt["coords"])
            if "level" in pt.keys():
                levels.append(pt["level"])
            
        lvl_max = np.max(levels)

        H = cp.Variable((self._dim, self._dim), symmetric = True)
        b = cp.Variable((self._dim,1))
        a = cp.Variable((1,1))

        Pquad = cp.bmat([ [a, b.T], [b, H] ])

        objective = cp.Maximize( cp.log_det(H) )
        constraints = [ Pquad >> 0 ]
        for vertex in minimum_bounding_rectangle(points):
            constraints += [ vertex.T @ H @ vertex + 2 * b.T @ vertex + a == 2*(lvl_max+self.constant) ]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        P = sp.linalg.block_diag(Pquad.value, np.zeros([self.kernel_dim-3,self.kernel_dim-3]))
        center = (- np.linalg.inv(H.value) @ b.value ).reshape(2)

        return P, center

    def function(self, point):
        '''
        Compute polynomial function numerically.
        '''
        m = self.kernel.function(point)
        return 0.5 * m.T @ self.matrix_coefs @ m - self.constant

    def gradient(self, point):
        '''
        Compute gradient of polynomial function numerically.
        '''
        m = self.kernel.function(point)
        Jac_m = self.kernel.jacobian(point)
        return Jac_m.T @ self.matrix_coefs @ m

    def hessian(self, point):
        '''
        Compute hessian of polynomial function numerically.
        '''
        m = self.kernel.function(point)
        Jac_m = self.kernel.jacobian(point)
        Hessian = np.zeros([self._dim, self._dim])
        for i in range(self._dim):
            for j in range(self._dim):
                Jac_m_i = Jac_m[:,i].reshape(self.kernel_dim)
                Jac_m_j = Jac_m[:,j].reshape(self.kernel_dim)
                hessian_m_ij = self.kernel._lambda_hessian_monomials(*point)[i][j].reshape(self.kernel_dim)
                Hessian[i,j] = Jac_m_i @ self.matrix_coefs @ Jac_m_j + m @ self.matrix_coefs @ hessian_m_ij
        return Hessian

    def plot_levels(self, levels, **kwargs):
        '''
        Modifies the level plot function for plotting with CLF/CBF colors
        '''
        if "colors" not in kwargs.keys():
            kwargs["colors"] = self.plot_config["color"]

        return super().plot_levels(levels, **kwargs)

    def get_shape(self):
        '''
        Return the polynomial coefficients.
        '''
        return self.matrix_coefs

    def get_kernel(self):
        '''
        Return the monomial basis vector.
        '''
        return self.kernel

    def __str__(self):
        type_fun = "Polynominal function ½ k(x)' P k(x)"
        if isinstance(self, KernelLyapunov):
            type_fun = "CLF ½ k(x)' P k(x)"
        if isinstance(self, KernelBarrier):
            type_fun = "CBF ½ ( k(x)' Q k(x) - 1 )"
        return type_fun

    # def SOS_convexity(self):
    #         '''
    #         Given a cvxpy P_var matrix and the function kernel, construct an efficient parametrization
    #         for SDP.
    #         '''
    #         n = self._dim
    #         p = self.kernel_dim
    #         A_list = self.kernel.get_A_matrices()

    #         y_alpha, _ = generate_monomial_list( self._dim, 1 )
    #         y_alpha = np.delete(y_alpha, 0, axis=0)

    #         augmented_alpha = np.array([ powers.tolist() + y_powers.tolist() for powers in self.kernel.alpha for y_powers in y_alpha ])
    #         # print( augmented_alpha )

    #         def add_monomial( monomials1, monomials2, current_alpha ):
    #             '''
    #             Adds monomial1 and/or monomial2 to monomial list, IFF 
    #             monomials1 and monomials2 cannot be made with the current alpha. 
    #             '''
    #             alpha_size = len(current_alpha)
    #             if alpha_size == 0:
    #                 current_alpha.append( monomials1.tolist() )

    #             # for i in range(alpha_size):
    #             #     for j in range(i,alpha_size):
    #             #         if monomials1 + monomials2 == np.array(current_alpha[i]) + np.array(current_alpha[j]):
    #             #             continue

    #             return current_alpha

    #         # Builds symbolic prototype for the SOS convex matrix
    #         current_alpha = []
    #         Prototype = sympy.MatrixSymbol('P', self.kernel_dim, self.kernel_dim)
    #         for i in range(n):
    #             for j in range(i,n):

    #                 # Inside each block matrix
    #                 Ai, Aj = A_list[i], A_list[j]
    #                 Block = Aj.T @ ( Ai.T @ Prototype + Prototype @ Ai ) + ( Ai.T @ Prototype + Prototype @ Ai ) @ Aj
    #                 NullElements = Block == 0

    #                 curr_row_alpha = augmented_alpha[augmented_alpha[:,n+i] == 1,:]
    #                 curr_col_alpha = augmented_alpha[augmented_alpha[:,n+j] == 1,:]

    #                 # print(curr_row_alpha)
    #                 # print(curr_col_alpha)

    #                 for k in range(p):
    #                     for l in range(k,p):

    #                         # Inside each element of the current block
    #                         if not NullElements[k,l]:
    #                             current_alpha = add_monomial( curr_row_alpha[k,:], curr_col_alpha[l,:], current_alpha )

class KernelLyapunov(KernelQuadratic):
    '''
    Class for kernel-based Lyapunov functions.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_param(self, param=None, **kwargs):
        '''
        Set the parameters of the Kernel Lyapunov function.
        Optional: pass a vector of parameters representing the vectorization of matrix P
        '''
        if param != None:
            super().set_param(coefficients=vector2sym(param))

        kwargs["constant"] = 0.0
        if "P" in kwargs.keys(): kwargs["coefficients"] = kwargs.pop("P")
        super().set_param(**kwargs)

    # def init(self):
    #     pass
    #             if isinstance(self, KernelLyapunov):
    #         '''
    #         If KernelLyapunov, make is SOS convex
    #         '''
    #         m_center = self.kernel.function(self.centers[0])
    #         Pquad = create_quadratic(eigen=[0.05, 0.05], R=rot2D(0.0), center=self.centers[0], kernel_dim=self.kernel_dim)
    #         # SOSConvexMatrix = cp.bmat([[ Aj.T @ ( Ai.T @ self.SHAPE + self.SHAPE @ Ai ) + ( Ai.T @ self.SHAPE + self.SHAPE @ Ai ) @ Aj for Aj in A_list ] for Ai in A_list ])
    #         constraints = [ self.SHAPE >> 0,
    #                         # SOSConvexMatrix >> 0,
    #                         m_center.T @ self.SHAPE @ m_center == 0 ]
    #         cost += cp.norm(self.SHAPE - Pquad)

    #         # for center in self.centers:
    #         #     self.point_list.append({"coords": center, "level": 0.0, "force": False})

    #     Computes the enclosing quadratic and adds a center point to point_list (the center of the quadratic)
    #     if isinstance(self, KernelBarrier):

    #         # Pquad, center_quad = self.enclosing_quadratic()
    #         # if len(self.centers) == 0:
    #         #     self.centers.append(center_quad)
    #         # else:

    #         for center in self.centers:
    #             self.point_list.append({"coords": center, "level":-self.constant})
    #             m = self.kernel.function(center)

    #         if self.fit_options["lower_bounded"]:
    #             constraints = [ F_var >> Pquad ]

    #         if self.fit_options["quadratic_like"]:
    #             cost = cp.norm(F_var - Pquad, 'fro')

class KernelBarrier(KernelQuadratic):
    '''
    Class for kernel-based barrier functions.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_param(self, param=None, **kwargs):
        '''
        Set the parameters of the Kernel Barrier function.
        Optional: pass a vector of parameters representing the vectorization of matrix Q
        '''
        if param != None:
            super().set_param(coefficients=vector2sym(param), constant=0.5)

        kwargs["constant"] = 0.5
        if "Q" in kwargs.keys(): kwargs["coefficients"] = kwargs.pop("Q")
        super().set_param(**kwargs)

        # Defines the CBF boundary
        if "boundary" in kwargs.keys():
            self.define_level_set(points=kwargs["boundary"], level=0.0, contained=True)
        
class KernelPair():
    '''
    Class for kernel-based CLF-CBF pairs
    '''
    def __init__(self, clf, cbf, plant, params = {"slack_gain": 1.0, "clf_gain": 1.0}):
        
        self.clf = clf
        self.cbf = cbf
        self.plant = plant
        self.params = params

        self.F = self.plant.get_F()
        self.P = self.clf.P
        self.Q = self.cbf.Q

        self.kernel = check_kernel(self.plant, self.clf, self.cbf)
        self.n = self.kernel._dim
        self.kernel_dim = self.kernel.kernel_dim
        self.A_list = self.kernel.get_A_matrices()

        self.plot_config = { "invariant_color": mcolors.BASE_COLORS['k'] }

    def det_invariant(self, xg, yg, extended=False):
        '''
        Evaluates det([ vecQ, vecP ]) = 0 over a grid.
        Parameters: xg, yg: (x,y) coords of each point in the grid
        Returns: a grid with the same size of xg, yg with the determinant values. 
        '''
        if xg.shape != yg.shape:
            raise Exception("x,y grid coordinates must have the same shape!")

        det_grid = np.zeros(xg.shape)
        for (i,j) in itertools.product(range( xg.shape[0] ), range( yg.shape[1] )):
            vecQ, vecP = np.zeros(self.n), np.zeros(self.n)
            z = self.kernel.function([xg[i,j], yg[i,j]])
            V = 0.5 * z.T @ self.P @ z
            for k in range(self.n):
                vecQ[k] = z.T @ self.A_list[k].T @ self.Q @ z
                vecP[k] = z.T @ self.A_list[k].T @ ( self.params["slack_gain"] * self.params["clf_gain"] * V * self.P - self.F ) @ z
            l = vecQ.T @ vecP / vecQ.T @ vecQ
            W = np.hstack([vecQ.reshape(self.n,1), vecP.reshape(self.n,1)])
            if l >= 0 or extended:
                # det_grid[i,j] = np.sqrt( np.linalg.det( W.T @ W ) ) # does not work
                det_grid[i,j] = np.linalg.det(W)
            else:
                det_grid[i,j] = np.inf

        return det_grid
    
    def invariant_set(self, limits, spacing=0.1, extended=False):
        '''
        Computes the invariant set for the given CLF-CBF pair
        '''
        x = np.arange(limits[0][0], limits[0][1],spacing)
        y = np.arange(limits[1][0], limits[1][1],spacing)
        xg, yg = np.meshgrid(x,y)

        invariant_contour = ctp.contour_generator(x=xg, y=yg, z=self.det_invariant(xg, yg, extended=extended) )
        self.invariant_segments = invariant_contour.lines(0.0)

    def plot_invariant(self, ax, limits, spacing=0.1, extended=False):
        '''
        Plots the invariant set segments into ax.
        '''
        self.invariant_set(limits, spacing=spacing, extended=extended)
        for seg in self.invariant_segments:
            ax.plot( seg[:,0], seg[:,1], color=self.plot_config["invariant_color"], linestyle='dashed', linewidth=1.0 )


class ApproxFunction(Function):
    '''
    Class for functions that can be approximated by a quadratic.
    '''
    def __init__(self, *args):
        super().__init__(*args)
        self.quadratic = Quadratic(*args)

    def compute_approx(self, value):
        '''
        Compute second order approximation for function.
        '''
        f = self.quadratic.evaluate_function(value)
        grad = self.quadratic.get_gradient()[0]
        H = self.quadratic.get_hessian()[0]
        inv_H = np.inv(H)

        l = np.sqrt( (2*f+1)/(grad.T @ inv_H @ grad) )
        v = l* inv_H @ grad

        self.quadratic.set_param(height = -0.5)
        self.quadratic.set_param(gradient = value - v)
        self.quadratic.set_param(hessian = H)

class CassiniOval(ApproxFunction):
    '''
    Class Cassini oval functions. Only works with 2-dimensional functions.
    '''
    def __init__(self, a, b, angle, *args):
        super().__init__(*args)
        self.a = a
        self.b = b
        self.e = self.a / self.b
        self.angle = math.degrees(angle)
        c, s = np.cos(self.angle), np.sin(self.angle)
        self.R = np.array([[c, -s],[s, c]])

    def function(self, point):
        '''
        2D Cassini oval function.
        '''
        v = self.R @ np.array(point)
        v1, v2 = v[0], v[1]
        return (v1**2 + v2**2)**2 - (2*self.a**2)*(v1**2 - v2**2) + self.a**4 - self.b**4

    def gradient(self, point):
        '''
        Gradient of the 2D Cassini oval function.
        '''
        v = self.R @ np.array(point)
        v1, v2 = v[0], v[1]

        grad_v = np.zeros(2)
        grad_v[0] = 4*( v1**2 + v2**2 )*v1 - (4*self.a**2)*v1
        grad_v[1] = 4*( v1**2 + v2**2 )*v2 + (4*self.a**2)*v2

        return grad_v.T @ self.R

    def hessian(self, point):
        '''
        Hessian of the 2D Cassini oval function.
        '''
        v = self.R @ np.array(point)
        v1, v2 = v[0], v[1]

        hessian_v = np.zeros([2,2])
        hessian_v[0,0] = 4*( 3*v1**2 + v2**2 ) - (4*self.a**2)
        hessian_v[1,1] = 4*( v1**2 + 3*v2**2 ) + (4*self.a**2)
        hessian_v[0,1] = 8*v1*v2
        hessian_v[1,0] = 8*v1*v2

        return hessian_v @ self.R

class CLBF(KernelQuadratic):
    '''
    Class for kernel-based Control Lyapunov Barrier Functions.
    '''
    def __init__(self, *args):
        super().__init__(*args)
        pass