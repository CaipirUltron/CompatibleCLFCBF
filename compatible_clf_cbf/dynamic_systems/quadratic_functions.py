import math
import numpy as np
from abc import ABC, abstractmethod


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
        if isinstance(value, list):
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

        self.set_param(**kwargs)

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


    # def superlevel(self, C, numpoints):
    #     '''
    #     This function returns the corresponding C-level set of the quadratic function, if its 2-dim.
    #     '''
    #     if self._dim != 2:
    #         return

    #     def change_variables(y1, y2):
    #         p = self.critical_point
    #         x = p[0] + Q[0,0]*y1 + Q[0,1]*y2
    #         y = p[1] + Q[1,0]*y1 + Q[1,1]*y2
    #         return x, y

    #     def parameterize_ellipse(delta):
    #         y1 = np.sqrt(delta/eig[0])*np.sin(t)
    #         y2 = np.sqrt(delta/eig[1])*np.cos(t)
    #         return y1, y2

    #         np.sqrt((2/np.abs(eigs[0])*np.abs(quadratic_level - height)))

    #     def parameterize_hyperbola(delta):
    #         # y1 = np.sqrt(np.abs(delta/eig[0]))*(1/np.cos(t))
    #         # y2 = np.sqrt(np.abs(delta/eig[1]))*np.tan(t)
    #         y1 = np.sqrt(np.abs(delta/eig[0]))*np.cosh(t)
    #         y2 = np.sqrt(np.abs(delta/eig[1]))*np.sinh(t)
    #         return y1, y2

    #     def parameterize(delta):
    #         if level_type == 'ellipse':
    #             y1, y2 = parameterize_ellipse(delta)
    #         elif level_type == 'hyperbola':
    #             y1, y2 = parameterize_hyperbola(delta)

    #         x, y = change_variables(y1, y2)
    #         u, v = np.zeros(numpoints), np.zeros(numpoints)
    #         for k in range(numpoints):
    #             self.evaluate([x[k],y[k]])
    #             gradient = self.get_gradient()
    #             u[k] = gradient[0]
    #             v[k] = gradient[1]

    #         return x, y, u, v

    #     t = np.linspace(-math.pi, math.pi, numpoints)

    #     # Parameterize 
    #     eig, Q = np.linalg.eig(self._hessian)
    #     if eig[0]*eig[1] > 0:
    #         # ellipse
    #         level_type = 'ellipse'
    #         if eig[0]>0:
    #             # convex
    #             delta = C-self.height
    #             if delta > 0:
    #                 x, y, u, v = parameterize(delta)
    #             else:
    #                 x, y, u, v = [], [], [], []
    #         else:
    #             # concave
    #             delta = self.height-C
    #             if delta > 0:
    #                 x, y, u, v = parameterize(delta)
    #             else:
    #                 x, y, u, v = [], [], [], []
    #     elif eig[0]*eig[1] < 0:
    #         # hyperbola
    #         level_type = 'hyperbola'
    #         if eig[0]>0:
    #             # convex
    #             delta = C-self.height
    #             if delta > 0:
    #                 x, y, u, v = parameterize(delta)
    #             else:
    #                 x, y, u, v = [], [], [], []
    #         else:
    #             # concave
    #             delta = self.height-C
    #             if delta > 0:
    #                 x, y, u, v = parameterize(delta)
    #             else:
    #                 x, y, u, v = [], [], [], []
    #     else:
    #         x, y, u, v = [], [], [], []

    #     return x, y, u, v

    @staticmethod
    def vector2sym(vector):
        '''
        Transforms numpy vector to corresponding symmetric matrix.
        '''
        dim = len(vector)
        if dim < 3:
            raise Exception("The input vector must be of length 3 or higher.")
        n = int((-1 + np.sqrt(1+8*dim))/2)
        sym_basis = Quadratic.symmetric_basis(n)
        M = np.zeros([n,n])
        for k in range(dim):
            M = M + sym_basis[k]*vector[k]
        return M

    @staticmethod
    def sym2vector(M):
        '''
        Stacks the cofficients of a symmetric matrix to a numpy vector.
        '''
        n = M.shape[0]
        if n < 2:
            raise Exception("The input matrix must be of size 2x2 or higher.")
        sym_basis = Quadratic.symmetric_basis(n)
        dim = int((n*(n+1))/2)
        vector = np.zeros(dim)
        for k in range(dim):
            list = np.nonzero(sym_basis[k])
            i, j = list[0][0], list[1][0]
            vector[k] = M[i][j]
        return vector

    @staticmethod
    def symmetric_basis(n):
        '''
        Returns the canonical basis of the space of symmetric (n x n) matrices.
        '''
        symm_basis = list()
        EYE = np.eye(n)
        for i in range(n):
            for j in range(i,n):
                if i == j:
                    symm_basis.append(np.outer(EYE[:,i], EYE[:,j]))
                else:
                    symm_basis.append(np.outer(EYE[:,i], EYE[:,j]) + np.outer(EYE[:,j], EYE[:,i]))
        return symm_basis

    @staticmethod
    def rot2D(theta):
        '''
        Standard 2D rotation matrix.
        '''
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s),(s,c)))
        return R

    @staticmethod
    def canonical2D(eigen, theta):
        '''
        Returns the (2x2) symmetric matrix with eigenvalues eigen and eigenvector angle theta.
        '''
        Diag = np.diag(eigen)
        R = Quadratic.rot2D(theta)
        H = R @ Diag @ R.T
        return H


class QuadraticLyapunov(Quadratic):
    '''
    Class for Quadratic Lyapunov functions.
    '''
    def __init__(self, init_value=0.0, **kwargs):
        Quadratic.__init__(self, init_value, **kwargs)
        self.set_param(height = 0.0)

class QuadraticBarrier(Quadratic):
    '''
    Class for Quadratic barrier functions.
    For positive definite Hessians, the unsafe set is described by the interior of an ellipsoid.
    '''
    def __init__(self, init_value=0.0, **kwargs):
        Quadratic.__init__(self, init_value, **kwargs)
        self.set_param(height = -0.5)