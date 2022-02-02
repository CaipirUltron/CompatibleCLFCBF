import math
import numpy as np
from abc import ABC, abstractmethod

from compatible_clf_cbf.dynamic_systems.dynamic_systems import Integrator
from compatible_clf_cbf.dynamic_systems.common_methods import vector2triangular, triangular2vector, sym2triangular, triangular_basis, sym2vector, vector2sym

class Function(ABC):
    '''
    Abstract implementation of general class for scalar functions.
    '''
    def __init__(self, init_value=0.0):
        self.set_value(init_value)
        self._function = 0.0
        if self._dim > 1:
            self._gradient = np.zeros(self._dim)
            self._hessian = np.zeros([self._dim,self._dim])
        else:
            self._gradient = 0.0
            self._hessian = 0.0

    def set_value(self, value):
        if isinstance(value, list) or isinstance(value,np.ndarray):
            self._dim = len(value)
            self._var = np.array(value)
        else:
            self._dim = 1
            self._var = value

    def compute(self):
        self.function()
        self.gradient()
        self.hessian()

    def evaluate(self, value):
        self.set_value(value)
        self.compute()
        return self._function

    def get_value(self):
        return self._var

    def get_fvalue(self):
        return self._function

    def get_gradient(self):
        return self._gradient

    def get_hessian(self):
        return self._hessian

    @abstractmethod
    def function(self):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def hessian(self):
        pass


class Quadratic(Function):
    '''
    Class for quadratic function representing x'Ax + b'x + c = 0.5 (x - p)'H(x-p) + height = 0.5 x'Hx - 0.5 p'(H + H')x + 0.5 p'Hp + height
    '''
    def __init__(self, init_value=0.0, **kwargs):

        # Set parameters
        super().__init__(init_value)

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

    def function(self):
        '''
        General quadratic function.
        '''
        self._function = self._var @ ( self.A @ self._var ) + self.b @ self._var + self.c

    def gradient(self):
        '''
        Gradient of general quadratic function.
        '''
        self._gradient = ( self.A + self.A.T ) @ self._var + self.b

    def hessian(self):
        '''
        Hessian of general quadratic function.
        '''
        self._hessian = ( self.A + self.A.T )

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
        eigen, Q = np.linalg.eig(self.get_hessian())
        angle = np.arctan2(Q[0, 1], Q[0, 0])
        return eigen, angle, Q

    def superlevel(self, level, num_points):
        '''
        This function returns the corresponding level set of the quadratic function, if its 2-dim.
        '''
        if self._dim != 2:
            Exception("Quadratic function is not two-dimensional.")

        eigs, Q = np.linalg.eig(self.get_hessian())
        height = self.get_height()

        scale_x = np.sqrt((2/np.abs(eigs[0])*np.abs(level - height)))
        scale_y = np.sqrt((2/np.abs(eigs[1])*np.abs(level - height)))

        def change_variables(y1, y2):
            p = self.critical_point
            x = p[0] + Q[0,0]*y1 + Q[0,1]*y2
            y = p[1] + Q[1,0]*y1 + Q[1,1]*y2
            return x, y

        def compute_grad_pts(x1, x2):
            u, v = np.zeros(num_points), np.zeros(num_points)
            for k in range(num_points):
                self.evaluate([x1[k],x2[k]])
                gradient = self.get_gradient()
                u[k], v[k] = gradient[0], gradient[1]
            return u, v

        t = np.linspace(-math.pi, math.pi, num_points)
        if eigs[0]*eigs[1] > 0:
            # ellipse
            y1, y2 = scale_x*np.cos(t), scale_y*np.sin(t)
            x1, x2 = change_variables( y1, y2 )
            u, v = compute_grad_pts( x1, x2 )

            x1, x2 = [x1, 0.0], [x2, 0.0]
            u, v = [u, 0.0], [v, 0.0]
        else:
            # hyperbola
            if eigs[0] < 0:
                hyper1_y1, hyper1_y2 = scale_x*np.sinh(t), scale_y*np.cosh(t)
                hyper2_y1, hyper2_y2 = scale_x*np.sinh(t), -scale_y*np.cosh(t)
            else:
                hyper1_y1, hyper1_y2 = scale_x*np.cosh(t), scale_y*np.sinh(t)
                hyper2_y1, hyper2_y2 = -scale_x*np.cosh(t), scale_y*np.sinh(t)

            hyper1_x1, hyper1_x2 = change_variables(hyper1_y1, hyper1_y2)
            hyper1_u, hyper1_v = compute_grad_pts( hyper1_x1, hyper1_x2 )

            hyper2_x1, hyper2_x2 = change_variables(hyper2_y1, hyper2_y2)
            hyper2_u, hyper2_v = compute_grad_pts( hyper2_x1, hyper2_x2 )

            x1 = [hyper1_x1, hyper2_x1]
            x2 = [hyper1_x2, hyper2_x2]

            u = [hyper1_u, hyper2_u]
            v = [hyper1_v, hyper2_v]

        return x1, x2, u, v


class QuadraticLyapunov(Quadratic):
    '''
    Class for Quadratic Lyapunov functions of the type (x-x0)'Hv(x-x0), parametrized by vector pi_v.
    Here, the Lyapunov minimum is a constant vector x0, and the hessian Hv is positive definite and parametrized by:
    Hv = Lv(pi_v)'Lv(pi_v) + epsilon I_n (Lv is upper triangular and epsilon is a small positive constant).
    '''
    def __init__(self, init_value=0.0, **kwargs):
        super().__init__(init_value, **kwargs)
        super().set_param(height=0.0)

        self.epsilon = 0.0
        self.Lv = sym2triangular( self.get_hessian()-self.epsilon*np.eye(self._dim) )
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
    Class for Quadratic barrier functions.
    For positive definite Hessians, the unsafe set is described by the interior of an ellipsoid.
    The symmetric Hessian is parametrized by Hh(pi) = \sum^n_i Li \pi_i, where {Li} is the canonical basis of the space of (n,n) symmetric matrices.
    '''
    def __init__(self, init_value=0.0, **kwargs):
        super().__init__(init_value, **kwargs)
        super().set_param(height = -0.5)

        self.param = sym2vector( self.get_hessian() )
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


class ApproxFunction(Function):
    '''
    Class for functions that can be approximated by a quadratic.
    '''
    def __init__(self, init_value=0.0):
        super().__init__(init_value)
        self.quadratic = Quadratic(init_value)

    def compute_approx(self, value):
        '''
        Compute second order approximation for function.
        '''
        f = self.quadratic.evaluate(value)
        grad = self.quadratic.get_gradient()
        H = self.quadratic.get_hessian()
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
    def __init__(self, a, b, angle, init_value=np.zeros(2)):
        super().__init__(init_value)
        self.a = a
        self.b = b
        self.e = self.a / self.b
        self.angle = math.degrees(angle)
        c, s = np.cos(self.angle), np.sin(self.angle)
        self.R = np.array([[c, -s],[s, c]])

    def function(self):
        '''
        2D Cassini oval function.
        '''
        v = self.R @ self._var
        v1, v2 = v[0], v[1]
        self._function = (v1**2 + v2**2)**2 - (2*self.a**2)*(v1**2 - v2**2) + self.a**4 - self.b**4

    def gradient(self):
        '''
        Gradient of the 2D Cassini oval function.
        '''
        v = self.R @ self._var
        v1, v2 = v[0], v[1]

        grad_v = np.zeros(2)
        grad_v[0] = 4*( v1**2 + v2**2 )*v1 - (4*self.a**2)*v1
        grad_v[1] = 4*( v1**2 + v2**2 )*v2 + (4*self.a**2)*v2

        self._gradient = grad_v.T @ self.R

    def hessian(self):
        '''
        Hessian of the 2D Cassini oval function.
        '''
        v = self.R @ self._var
        v1, v2 = v[0], v[1]

        hessian_v = np.zeros([2,2])
        hessian_v[0,0] = 4*( 3*v1**2 + v2**2 ) - (4*self.a**2)
        hessian_v[1,1] = 4*( v1**2 + 3*v2**2 ) + (4*self.a**2)
        hessian_v[0,1] = 8*v1*v2
        hessian_v[1,0] = 8*v1*v2

        self._hessian = hessian_v @ self.R