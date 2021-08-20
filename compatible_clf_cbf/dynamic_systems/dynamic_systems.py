import numpy as np
from sage.all import *
from sage.symbolic.function import BuiltinFunction


class NonlinearSystem(BuiltinFunction):
    '''
    Class for nonlinear system plant.
    '''
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
    '''
    Class for affine plant.
    '''
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
    '''
    Class for quadratic function representing x'Ax + b'x + c
                                            = 0.5 (x - p)'H(x-p) + height
                                            = 0.5 x'Hx - 0.5 p'(H + H')x + 0.5 p'Hp + height
    '''
    def __init__(self, var_str, **kwargs):
        BuiltinFunction.__init__(self, 'Quadratic Function', nargs=1)

        self._var = vector(var(var_str+','))
        self._dimension = np.size(self._var)

        # Set parameters
        self.A = np.zeros([self._dimension,self._dimension])
        self.b = np.zeros(self._dimension)
        self.c = 0.0

        self.hessian_matrix = np.zeros([self._dimension,self._dimension])
        self.critical_point = np.zeros(self._dimension)
        self.height = 0.0

        self.set_param(**kwargs)

        # Set eigenbasis for hessian matrix
        _, _, Q = self.compute_eig()
        self.eigen_basis = np.zeros([self._dimension, self._dimension, self._dimension])
        for k in range(self._dimension):
            self.eigen_basis[:][:][k] = np.outer( Q[:,k], Q[:,k] )

        self._gen_var_str()
        self._create_dictionary()

    def set_param(self, **kwargs):
        '''
        Sets the Quadratic function parameters.
        '''
        for key in kwargs:
            if key == "hessian":
                self.hessian_matrix = kwargs[key]
            if key == "critical":
                self.critical_point = kwargs[key]
            if key == "height":
                self.height = kwargs[key]

        self.A = 0.5 * self.hessian_matrix
        self.b = - 0.5*( self.hessian_matrix + self.hessian_matrix.T ).dot( self.critical_point )
        self.c = 0.5 * self.critical_point.dot( self.hessian_matrix.dot(self.critical_point) ) + self.height

        for key in kwargs:
            if key == "A":
                self.A = kwargs[key]
            if key == "b":
                self.b = kwargs[key]
            if key == "c":
                self.c = kwargs[key]

        A_symb, b_symb = matrix(self.A), vector(self.b)
        self._function = self._var * ( A_symb * self._var ) + b_symb * self._var + self.c
        self._gradient = self._function.gradient()
        self._hessian = self._function.hessian()

    def eigen2hessian(self, eigen):
        '''
        Returns hessian matrix from a given set of eigenvalues
        '''
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

    def critical(self):
        return self.critical_point

    def compute_eig(self):
        eigen, Q = np.linalg.eig(self.hessian())
        angle = np.arctan2(Q[0, 1], Q[0, 0])
        return eigen, angle, Q

    def superlevel(self, C, numpoints):
        '''
        This function returns the corresponding C-level set of the quadratic function, if its 2-dim.
        '''
        if self._dimension != 2:
            return

        def change_variables(y1, y2):
            p = self.critical_point
            x = p[0] + Q[0,0]*y1 + Q[0,1]*y2
            y = p[1] + Q[1,0]*y1 + Q[1,1]*y2
            return x, y

        def parameterize_ellipse(delta):
            y1 = np.sqrt(delta/eig[0])*np.sin(t)
            y2 = np.sqrt(delta/eig[1])*np.cos(t)
            return y1, y2

        def parameterize_hyperbola(delta):
            # y1 = np.sqrt(np.abs(delta/eig[0]))*(1/np.cos(t))
            # y2 = np.sqrt(np.abs(delta/eig[1]))*np.tan(t)
            y1 = np.sqrt(np.abs(delta/eig[0]))*np.cosh(t)
            y2 = np.sqrt(np.abs(delta/eig[1]))*np.sinh(t)
            return y1, y2

        def parameterize(delta):
            if level_type == 'ellipse':
                y1, y2 = parameterize_ellipse(delta)
            elif level_type == 'hyperbola':
                y1, y2 = parameterize_hyperbola(delta)

            x, y = change_variables(y1, y2)
            u, v = np.zeros(numpoints), np.zeros(numpoints)
            for k in range(numpoints):
                gradient = self.gradient(np.array([x[k],y[k]]))
                u[k] = gradient[0]
                v[k] = gradient[1]

            return x, y, u, v

        t = np.linspace(-math.pi, math.pi, numpoints)

        # Parameterize 
        eig, Q = np.linalg.eig(self.hessian_matrix)
        if eig[0]*eig[1] > 0:
            # ellipse
            level_type = 'ellipse'
            if eig[0]>0:
                # convex
                delta = C-self.height
                if delta > 0:
                    x, y, u, v = parameterize(delta)
                else:
                    x, y, u, v = [], [], [], []
            else:
                # concave
                delta = self.height-C
                if delta > 0:
                    x, y, u, v = parameterize(delta)
                else:
                    x, y, u, v = [], [], [], []
        elif eig[0]*eig[1] < 0:
            # hyperbola
            level_type = 'hyperbola'
            if eig[0]>0:
                # convex
                delta = C-self.height
                if delta > 0:
                    x, y, u, v = parameterize(delta)
                else:
                    x, y, u, v = [], [], [], []
            else:
                # concave
                delta = self.height-C
                if delta > 0:
                    x, y, u, v = parameterize(delta)
                else:
                    x, y, u, v = [], [], [], []
        else:
            x, y, u, v = [], [], [], []

        return x, y, u, v

    def _eval_numpy_(self, vars):
        for i in range(self._dimension):
            self._var_dictionary[ self._var_str[i] ] = vars[i]
        return self._function(**self._var_dictionary)

    @staticmethod
    def vector2sym(vector):
        '''
        Transforms numpy vector to corresponding symmetric matrix.
        '''
        dim = vector.shape[0]
        if dim < 3:
            raise Exception("The input vector must be of length 3 or higher.")
        n = int((-1 + np.sqrt(1+8*dim))/2)
        sym_basis = QuadraticFunction.symmetric_basis(n)
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
        sym_basis = QuadraticFunction.symmetric_basis(n)
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
        R = QuadraticFunction.rot2D(theta)
        H = np.matmul(R, np.matmul(Diag, R.T))
        return H


class QuadraticLyapunov(QuadraticFunction):
    '''
    Class for Quadratic Lyapunov functions.
    '''
    def __init__(self, var_str, **kwargs):
        QuadraticFunction.__init__(self, var_str, **kwargs)


class QuadraticBarrier(QuadraticFunction):
    '''
    Class for Quadratic barrier functions.
    For positive definite Hessians, the unsafe set is described by the interior of an ellipsoid.
    '''
    def __init__(self, var_str, **kwargs):
        QuadraticFunction.__init__(self, var_str, **kwargs)
        self.set_param(height = -0.5)