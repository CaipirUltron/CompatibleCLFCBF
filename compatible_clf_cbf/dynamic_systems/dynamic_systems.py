import numpy as np
from sage.all import *
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

    def __init__(self, var_str, A, b, c):
        BuiltinFunction.__init__(self, 'Quadratic Function', nargs=1)

        self._var = vector(var(var_str+','))
        self._dimension = np.size(self._var)

        self.set_param(A,b,c)

        self._gen_var_str()
        self._create_dictionary()

    def set_param(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c

        A_symb, b_symb = matrix(self.A), vector(self.b)
        self._function = self._var * ( A_symb * self._var ) + b_symb * self._var + self.c
        self._gradient = self._function.gradient()
        self._hessian = self._function.hessian()

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
        clf_lambda, clf_rot = np.linalg.eig(self.A)
        clf_angle = np.arctan2(clf_rot[0, 1], clf_rot[0, 0])
        return clf_lambda, clf_angle

    def _eval_numpy_(self, vars):
        for i in range(self._dimension):
            self._var_dictionary[ self._var_str[i] ] = vars[i]

        return self._function(**self._var_dictionary)


class QuadraticLyapunov(QuadraticFunction):

    def __init__(self, var_str, H, x0):

        self.minimum = x0
        self.A = H
        self.b = -2*self.A.T.dot(self.minimum)
        self.c = self.A.dot(self.minimum).dot(self.minimum)
        
        QuadraticFunction.__init__(self, var_str, self.A, self.b, self.c)

    def set_hessian(self, H):
        A = H
        b = -2*self.A.T.dot(self.minimum)
        c = self.A.dot(self.minimum).dot(self.minimum)
        self.set_param(A,b,c)


class QuadraticBarrier(QuadraticFunction):

    def __init__(self, var_str, H, x0):

        self.minimum = x0
        self.A = H
        self.b = -2*self.A.T.dot(self.minimum)
        self.c = self.A.dot(self.minimum).dot(self.minimum) - 1
        
        QuadraticFunction.__init__(self, var_str, self.A, self.b, self.c)

    def set_hessian(self, H):
        A = H
        b = -2*self.A.T.dot(self.minimum)
        c = self.A.dot(self.minimum).dot(self.minimum) - 1
        self.set_param(A,b,c)