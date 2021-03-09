import numpy as np
from numpy.core.function_base import linspace
from sage.all import *
from sage.combinat.sloane_functions import A109814
from sage.symbolic.function import BuiltinFunction
from sage.calculus.var import *


class NonlinearSystem(BuiltinFunction):
    
    def __init__(self, state_str, control_str, *expressions):
        BuiltinFunction.__init__(self, 'Dynamic System', nargs=2)

        self._state = vector(var(state_str+','))
        self._control = vector(var(control_str+','))

        self.state_dim = np.size(self._state)
        self.control_dim = np.size(self._control)

        self._gen_state_str()
        self._gen_control_str()
        self._create_dictionary()

        self.set_expression(*expressions)

    def _gen_state_str(self):
        self._state_str = list()
        for i in range(self.state_dim):
            self._state_str.append(str(self._state[i]))

    def _gen_control_str(self):
        self._control_str = list()
        for i in range(self.control_dim):
            self._control_str.append(str(self._control[i]))

    def _create_dictionary(self):
        self._var_dictionary = {}
        for i in range(self.state_dim):
            self._var_dictionary[self._state_str[i]] = 0.0
        for j in range(self.control_dim):
            self._var_dictionary[self._control_str[j]] = 0.0

    def set_expression(self, *expressions):
        f_array = []
        for f_expr in expressions:
            f_array.append(SR(f_expr))
        self._vector_field = vector(f_array)

    def state(self):
        return self._state

    def control(self):
        return self._control

    def expression(self):
        return self._vector_field

    def _eval_numpy_(self, State, Control):
        for i in range(self.state_dim):
            self._var_dictionary[ self._state_str[i] ] = State[i]
        for j in range(self.control_dim):
            self._var_dictionary[ self._control_str[j] ] = Control[j]

        return self._vector_field(**self._var_dictionary)


class AffineSystem(NonlinearSystem):

    def __init__(self, state_str, control_str, f, *columns_g):

        NonlinearSystem.__init__(self, state_str, control_str)

        f_array = []
        for i in range(len(f)):
            f_array.append(SR(f[i]))
        self._f = vector(f_array)

        g_array_transposed = []
        for column_g in columns_g:
            g_array_transposed.append(column_g)
            for i in range(len(column_g)):
                g_array_transposed[-1][i] = SR(column_g[i])
        g_array = np.array(g_array_transposed).T.tolist()
        self._g = matrix(g_array)

        self._vector_field = self._f + self._g * self._control

    def compute_f(self, state):
        for i in range(self.state_dim):
            self._var_dictionary[ self._state_str[i] ] = state[i]
        return np.array(self._f(**self._var_dictionary),dtype=float)

    def compute_g(self, state):
        for i in range(self.state_dim):
            self._var_dictionary[ self._state_str[i] ] = state[i]
        return np.array(self._g(**self._var_dictionary),dtype=float)

class LinearSystem(NonlinearSystem):

    def __init__(self, state_str, control_str, A, B):
        NonlinearSystem.__init__(self, state_str, control_str)

        A_symb, B_symb = matrix(A), matrix(B)
        self._f = A_symb * self._state + B_symb * self._control


class QuadraticFunction(BuiltinFunction):

    def __init__(self, var_str, A, b, c, *types):
        BuiltinFunction.__init__(self, 'Quadratic Function', nargs=1)

        self._var = vector(var(var_str+','))
        self._dimension = np.size(self._var)

        # Configure type: 'default' = x'Ax + b'x + c
        #                 'factored' = (x - b)'A(x-b) + c
        if len(types) == 0:
            self.type = 'default'
        else:
            for type in types:
                if type == 'default':
                    self.type = 'default'
                elif type == 'factored':
                    self.type = 'factored'

        # Set parameters
        self.set_param(A,b,c)

        # Set eigenbasis for hessian matrix
        _, _, Q = self.compute_eig()
        self.eigen_basis = np.zeros([self._dimension, self._dimension, self._dimension])
        for k in range(self._dimension):
            self.eigen_basis[:][:][k] = np.outer( Q[:,k], Q[:,k] )

        self._gen_var_str()
        self._create_dictionary()

    def set_param(self, A, b, c):

        if self.type == 'default':
            self.A = A
            self.b = b
            self.c = c
        else:
            self.hessian_matrix = A
            self.critical_point = b
            self.height = c
            self.A = self.hessian_matrix
            self.b = -(self.hessian_matrix + self.hessian_matrix.T).dot(self.critical_point)
            self.c = self.height + self.hessian_matrix.dot(self.critical_point).dot(self.critical_point)

        A_symb, b_symb = matrix(self.A), vector(self.b)
        self._function = self._var * ( A_symb * self._var ) + b_symb * self._var + self.c
        self._gradient = self._function.gradient()
        self._hessian = self._function.hessian()

    # Define A, b, c for Quadratics of type 'default' = x'Ax + b'x + c
    def set_A(self, A):
        if self.type == 'factored':
            return
        else:
            self.set_param(A, self.b, self.c)

    def set_b(self, b):
        if self.type == 'factored':
            return
        else:
            self.set_param(self.A, b, self.c)

    def set_c(self, c):
        if self.type == 'factored':
            return
        else:
            self.set_param(self.A, self.b, c)

    # Define hessian, critical point and height for Quadratics of type 'factored' = (x - b)'A(x-b) + c
    def set_hessian(self, H):
        if self.type == 'default':
            return
        else:
            self.set_param(H, self.critical_point, self.height)

    def set_critical(self, x0):
        if self.type == 'default':
            return
        else:
            self.set_param(self.hessian_matrix, x0, self.height)

    def set_height(self, height):
        if self.type == 'default':
            return
        else:
            self.set_param(self.hessian_matrix, self.critical_point, height)

    # Returns hessian matrix from a given set of eigenvalues
    def eigen2hessian(self, eigen):
        if self._dimension != len(eigen):
                raise Exception("Dimension mismatch.")

        H = np.zeros(self._dimension)
        for k in range(self._dimension):
            H = H + eigen[k] * self.eigen_basis[:][:][k]

        return H

    def _gen_var_str(self):
        self._var_str = list()
        for i in range(self._dimension):
            self._var_str.append(str(self._var[i]))

    def _create_dictionary(self):
        self._var_dictionary = {}
        for i in range(self._dimension):
            self._var_dictionary[self._var_str[i]] = []

    def function(self, *vars):
        if len(vars) == 0:
            return self._function
        else:
            for i in range(self._dimension):
                self._var_dictionary[ self._var_str[i] ] = vars[0][i]
            return self._function(**self._var_dictionary)

    def gradient(self, *vars):
        if len(vars) == 0:
            return self._gradient
        else:
            for i in range(self._dimension):
                self._var_dictionary[ self._var_str[i] ] = vars[0][i]
            return np.array(self._gradient(**self._var_dictionary),dtype=float)

    def hessian(self):
        return np.array(self._hessian, dtype=float)

    def compute_eig(self):
        eigen, Q = np.linalg.eig(self.hessian_matrix)
        angle = np.arctan2(Q[0, 1], Q[0, 0])
        return eigen, angle, Q

    # This function returns the corresponding C-level set of the quadratic function, if its 2-dim
    def superlevel(self, C, numpoints):

        if self._dimension != 2:
            return

        def parameterize_ellipse(delta):

            y1 = np.sqrt(delta/eig[0])*np.sin(t)
            y2 = np.sqrt(delta/eig[1])*np.cos(t)

            p = self.critical_point
            x = p[0] + Q[0,0]*y1 + Q[0,1]*y2
            y = p[1] + Q[1,0]*y1 + Q[1,1]*y2

            return x, y

        t = np.linspace(0, 2*math.pi, numpoints)

        eig, Q = np.linalg.eig(self.hessian_matrix)
        if eig[0]*eig[1] > 0:
            # ellipse
            if eig[0]>0:
                # convex
                delta = C-self.height
                if delta > 0:
                    x, y = parameterize_ellipse(delta)
                else:
                    x, y = [], []
            else:
                # concave
                delta = self.height-C
                if delta > 0:
                    x, y = parameterize_ellipse(delta)
                else:
                    x, y = [], []
        elif eig[0]*eig[1] < 0:
            # hyperbola
            x, y = [], []
        else:
            x, y = [], []

        return x, y

    def _eval_numpy_(self, vars):
        for i in range(self._dimension):
            self._var_dictionary[ self._var_str[i] ] = vars[i]

        return self._function(**self._var_dictionary)
 
    @staticmethod
    def symmetric_basis(n):

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
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s),(s,c)))
        return R

    @staticmethod
    def canonical2D(eigen, theta):
        Diag = np.diag(eigen)
        R = QuadraticFunction.rot2D(theta)
        H = np.matmul(R, np.matmul(Diag, R.T))
        return H


class QuadraticLyapunov(QuadraticFunction):

    def __init__(self, var_str, H, x0):
        QuadraticFunction.__init__(self, var_str, H, x0, 0.0, 'factored')


class QuadraticBarrier(QuadraticFunction):

    def __init__(self, var_str, H, x0):
        QuadraticFunction.__init__(self, var_str, H, x0, -1.0, 'factored')