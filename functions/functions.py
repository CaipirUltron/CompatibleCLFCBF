import math
import numpy as np
import sympy as sp

from copy import copy
from common import *
from dynamic_systems import Integrator


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
        return self._function

    def get_gradient(self):
        return self._gradient

    def get_hessian(self):
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

        function = copy(self)
        function.functions.append(func)

        return function

    def contour_plot(self, ax, levels, colors, min=-10.0, max=10.0, resolution=0.1):
        '''
        Return 2D contour plot object.
        '''
        if self._dim != 2:
            raise Exception("Contour plot can only be used for 2D functions.")

        x = np.arange(min, max, resolution)
        y = np.arange(min, max, resolution)
        xv, yv = np.meshgrid(x,y)

        mesh_fvalues = np.zeros([np.size(xv,0),np.size(xv,1)])
        for i in range(np.size(xv,1)):
            mesh_fvalues[:,i] = np.array(self.evaluate_function(xv[:,i], yv[:,i]))

        cs = ax.contour(xv, yv, mesh_fvalues, levels=levels, colors=colors)
        return cs


class Quadratic(Function):
    '''
    Class for quadratic function representing x'Ax + b'x + c = 0.5 (x - p)'H(x-p) + height = 0.5 x'Hx - 0.5 p'(H + H')x + 0.5 p'Hp + height
    '''
    def __init__(self, *args, **kwargs):

        # Set parameters
        super().__init__(*args)

        if self._dim > 1:
            self.A = np.zeros([self._dim,self._dim])
            self.b = np.zeros(self._dim)
            self.critical_point = np.zeros(self._dim)
        else:
            self.A = 0.0
            self.b = 0.0
            self.critical_point = 0.0
        self.c = 0.0
        self.height = 0.0

        Quadratic.set_param(self, **kwargs)

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
        super().__init__(*args, **kwargs)
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
        super().__init__(*args, **kwargs)
        super().set_param(height = -0.5)

        self.param = sym2vector( self._hessian )
        self.dynamics = Integrator(self.param,np.zeros(len(self.param)))

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

    def update(self, param_ctrl, dt):
        '''
        Integrates the barrier function parameters.
        '''
        self.dynamics.set_control(param_ctrl)
        self.dynamics.actuate(dt)
        self.set_param(self.dynamics.get_state())


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


class PolynomialFunction(Function):
    '''
    Class for polynomial functions of the type m(x)' P m(x) of maximum degree 2*d, where m(x) is a vector of (n+d,d) known monomials.
    '''
    def __init__(self, *args, **kwargs):

        # Initialization
        super().__init__(*args)
        self.set_param(**kwargs)

        # Create symbols
        self._symbols = []
        for dim in range(self._dim):
            self._symbols.append( sp.Symbol('x' + str(dim+1)) )

        # Generate monomial list and symbolic monomials
        self.alpha = generate_monomial_list( self._dim, self._degree )
        self._monomials = generate_monomials_from_symbols( self._symbols, self._degree )
        self._num_monomials = len(self._monomials)

        # Symbolic computations
        self._symP = sp.Matrix(sp.symarray('p',(self._num_monomials,self._num_monomials)))
        self._sym_monomials = sp.Matrix(self._monomials)
        self._sym_jacobian_monomials = self._sym_monomials.jacobian(self._symbols)

        self._hessian_monomials = [ [0 for i in range(self._dim)] for j in range(self._dim) ]
        for i in range(self._dim):
            for j in range(self._dim):
                self._hessian_monomials[i][j] = sp.diff(self._sym_jacobian_monomials[:,j], self._symbols[i])

        self._sym_fun = ( self._symP * self._sym_monomials ).dot(self._sym_monomials)

        # Compute numeric A matrices
        self.compute_A()

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
        self._coefficients = np.zeros([self._num_monomials, self._num_monomials])

        for key in kwargs:
            if key == "degree":
                self._degree = kwargs[key]
                self._num_monomials = num_comb(self._dim, self._degree)
                self._coefficients = np.zeros([self._num_monomials, self._num_monomials])
            if key == "P":
                self._coefficients = np.array(kwargs[key])

        if np.shape(self._coefficients) != (self._num_monomials, self._num_monomials):
            raise Exception("P must be (N x N), where N is the dimension of the monomial basis!")

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

    def function(self, point):
        '''
        Compute polynomial function numerically.
        '''
        m = np.array(self._lambda_monomials(*point))
        return m @ self._coefficients @ m

    def gradient(self, point):
        '''
        Compute gradient of polynomial function numerically.
        '''
        m = np.array(self._lambda_monomials(*point))
        del_m = np.array(self._lambda_jacobian_monomials(*point))
        return m @ self._coefficients @ del_m + del_m.T @ self._coefficients @ m

    def hessian(self, point):
        '''
        Compute hessian of polynomial function numerically.
        '''
        m = np.array(self._lambda_monomials(*point))
        Hv = np.zeros([self._dim, self._dim])
        for i in range(self._dim):
            for j in range(self._dim):
                delm_i = np.array(self._lambda_jacobian_monomials(*point))[:,i].reshape(self._num_monomials)
                delm_j = np.array(self._lambda_jacobian_monomials(*point))[:,j].reshape(self._num_monomials)
                hessian_m_ij = self._lambda_hessian_monomials(*point)[i][j].reshape(self._num_monomials)
                Hv[i,j] = delm_i @ ( self._coefficients + self._coefficients.T ) @ delm_j + m @ ( self._coefficients + self._coefficients.T ) @ hessian_m_ij
        return Hv

    def get_matrices(self):
        '''
        Return the A matrices.
        '''
        return self.A

    def get_symbols(self):
        '''
        Return the sympy variables.
        '''
        return self._symbols

    def get_monomials(self):
        '''
        Return the monomial basis vector.
        '''
        return self._monomials

    def get_polynomial(self):
        '''
        Return the complete symbolic polynomial of the function
        '''
        return self._sym_fun

    def get_degree(self):
        '''
        Return the maximum polynomial degree. 
        '''
        return self._degree


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


# class CLBF(Function):
#     '''
#     Class for Gaussian-based Control Lyapunov Barrier Functions.
#     '''
#     def __init__(self, *args, goal = Gaussian(), obstacles = []):
#         super().__init__(*args)
#         self.set_goal(goal)
#         self.dim = self.goal_gaussian._dim
#         self.obstacle_gaussians = []
#         for obs in obstacles:
#             if obs._dim != self.dim:
#                 raise Exception("Dimension of goal Gaussian and obstacle Gaussian must be the same.")
#             self.add_obstacle(obs)

#     def set_goal(self, goal):
#         self.goal_gaussian = goal

#     def add_obstacle(self, obstacle):
#         self.obstacle_gaussians.append(obstacle)

#     def function(self, point):
#         '''
#         Gaussian CLBF function.
#         '''
#         self.goal_gaussian.set_value(*point)
#         self.goal_gaussian.function()

#         sum_obs_gaussians = 0.0
#         self.goal_gaussian.compute()
#         for obs in self.obstacle_gaussians:
#             obs.set_value(np.array(point))
#             obs.function()
#             sum_obs_gaussians += obs.get_function()
#         self._function = - self.goal_gaussian.get_function() + sum_obs_gaussians

#     def gradient(self, point):
#         '''
#         Gradient of Gaussian CLBF function.
#         '''
#         self.goal_gaussian.set_value(np.array(point))
#         self.goal_gaussian.gradient()

#         sum_obs_gradients =  np.zeros(self.dim)
#         for obs in self.obstacle_gaussians:
#             obs.set_value(np.array(point))
#             obs.gradient()
#             sum_obs_gradients += obs.get_gradient()
#         self._gradient = - self.goal_gaussian.get_gradient() + sum_obs_gradients

#     def hessian(self):
#         '''
#         Hessian of Gaussian CLBF function.
#         '''
#         self.goal_gaussian.set_value(self._var)
#         self.goal_gaussian.hessian()

#         sum_obs_hessians =  np.zeros([self.dim,self.dim])
#         for obs in self.obstacle_gaussians:
#             obs.set_value(self._var)
#             obs.hessian()
#             sum_obs_hessians += obs.get_hessian()
#         self._hessian = - self.goal_gaussian.get_hessian() + sum_obs_hessians