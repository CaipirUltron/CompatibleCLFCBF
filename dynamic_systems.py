import numpy as np
from sage.all import *
from sage.symbolic.function import BuiltinFunction
from sage.calculus.var import *

class NonlinearSystem(BuiltinFunction):

    def __init__(self, state_str, control_str, *f_expressions):
        BuiltinFunction.__init__(self, 'Dynamic System', nargs=2)

        self._state = vector(var(state_str+','))
        self._control = vector(var(control_str+','))

        self._state_dim = np.size(self._state)
        self._control_dim = np.size(self._control)

        self._gen_state_str()
        self._gen_control_str()
        self._create_dictionary()

        self.change_expression(*f_expressions)

    def _create_dictionary(self):
        self._var_dictionary = {}
        for i in range(self._state_dim):
            self._var_dictionary[self._state_str[i]] = []
        for j in range(self._control_dim):
            self._var_dictionary[self._control_str[j]] = []

    def change_expression(self, *f_expressions):
        f_array = []
        for f_expr in f_expressions:
            f_array.append(SR(f_expr))
        self._f = vector(f_array)

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
        return self._f

    def _eval_numpy_(self, State, Control):

        for i in range(self._state_dim):
            self._var_dictionary[ self._state_str[i] ] = State[i]
        for j in range(self._control_dim):
            self._var_dictionary[ self._control_str[j] ] = Control[j]

        return self._f(**self._var_dictionary)

    # def _gen_state(self):
    #     self._state_str = 'x1'
    #     for i in range(self._state_dim-1):
    #         self._state_str += (' x'+str(i+2))

    # def _gen_control(self):
    #     self._control_str = 'u1'
    #     for i in range(self._control_dim-1):
    #         self._control_str += (' u'+str(i+2))


class LinearSystem(NonlinearSystem):
    def __init__(self, state_str, control_str, A, B):
        NonlinearSystem.__init__(self, state_str, control_str)

        Asym = matrix(A)
        Bsym = matrix(B)
        self._f = Asym * self._state + Bsym * self._control