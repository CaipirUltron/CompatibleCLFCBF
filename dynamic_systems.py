import numpy as np
from sage.all import *
from sage.symbolic.function import BuiltinFunction
from sage.calculus.var import *

class NonlinearSystem(BuiltinFunction):

    def __init__(self, state_str, control_str, *expressions):
        BuiltinFunction.__init__(self, 'Dynamic System', nargs=2)

        self._state = vector(var(state_str+','))
        self._control = vector(var(control_str+','))

        self._state_dim = np.size(self._state)
        self._control_dim = np.size(self._control)

        self._gen_state_str()
        self._gen_control_str()
        self._create_dictionary()

        self.change_expression(*expressions)

    def _create_dictionary(self):
        self._var_dictionary = {}
        for i in range(self._state_dim):
            self._var_dictionary[self._state_str[i]] = []
        for j in range(self._control_dim):
            self._var_dictionary[self._control_str[j]] = []

    def change_expression(self, *expressions):
        f_array = []
        for f_expr in expressions:
            f_array.append(SR(f_expr))
        self._vector_field = vector(f_array)

    def _gen_state_str(self):
        self._state_str = list()
        for i in range(self._state_dim):
            self._state_str.append(str(self._state[i]))

    def _gen_control_str(self):
        self._control_str = list()
        for i in range(self._control_dim):
            self._control_str.append(str(self._control[i]))

    def state(self):
        return self._state

    def control(self):
        return self._control

    def expression(self):
        return self._vector_field

    def _eval_numpy_(self, State, Control):

        for i in range(self._state_dim):
            self._var_dictionary[ self._state_str[i] ] = State[i]
        for j in range(self._control_dim):
            self._var_dictionary[ self._control_str[j] ] = Control[j]

        return self._vector_field(**self._var_dictionary)

class LinearSystem(NonlinearSystem):
    def __init__(self, state_str, control_str, A, B):
        NonlinearSystem.__init__(self, state_str, control_str)

        Asym, Bsym = matrix(A), matrix(B)
        self._f = Asym * self._state + Bsym * self._control

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