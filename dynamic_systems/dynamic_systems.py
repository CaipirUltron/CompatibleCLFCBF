import scipy
import numpy as np

from abc import ABC, abstractmethod

class DynamicSystem(ABC):
    '''
    Abstract class for dynamic systems. This class has all the functionality for simulating dynamic systems using scipy integration methods.
    The abstract functionality that needs to be implemented by the child classes is the flow computation.
    '''
    def __init__(self, initial_state, initial_control):
        
        # Sets integration method from scipy
        self.n = len(initial_state)
        self.m = len(initial_control)

        from scipy.integrate import ode
        self.mODE = ode(self.get_flow).set_integrator('dopri5')

        self.state_log = []
        for _ in range(0, self.n):
            self.state_log.append([])

        self.control_log = []
        for _ in range(0, self.m):
            self.control_log.append([])

        self.set_state(initial_state)
        self.set_control(initial_control)

    def set_state(self, state):
        '''
        Sets system state.
        '''
        if type(state) == list or type(state) == np.ndarray:
            self._state = np.array(state)
        else:
            self._state = np.array([state])

        self._dstate = np.zeros(self.n)
        self.mODE.set_initial_value(self._state)
        # self.log_state()

    def set_control(self, control_input):
        '''
        Sets system control.
        '''
        if type(control_input) == list or type(control_input) == np.ndarray:
            self._control = np.array(control_input)
        else:
            self._control = np.array([control_input])
        # self.log_control()

    def actuate(self, dt):
        '''
        Sends the control inputs.
        '''
        self.dynamics()
        self._state = self.mODE.integrate(self.mODE.t+dt)
        self.log_state()
        self.log_control()

    def log_state(self):
        '''
        Logs the state.
        '''
        for state_dim in range(0, self.n):
            self.state_log[state_dim].append(self._state[state_dim])

    def log_control(self):
        '''
        Logs the control.
        '''
        for ctrl_dim in range(0, self.m):
            self.control_log[ctrl_dim].append(self._control[ctrl_dim])

    def get_flow(self, t):
        '''
        Gets the current system flow, or state derivative.
        '''
        return self._dstate

    def get_state(self):
        '''
        Gets the current system state.
        '''
        return self._state

    def get_control(self):
        '''
        Gets the last control input.
        '''
        return self._control

    @abstractmethod
    def dynamics(self):
        pass

class AffineSystem(DynamicSystem):
    '''
    General class for an affine system dx = f(x) + g(x) u.
    '''
    def __init__(self, initial_state, initial_control):
        super().__init__(initial_state, initial_control)
        self._f = np.zeros(self.n)
        self._g = np.zeros([self.n, self.m])

    def get_f(self):
        '''
        Gets the value of f(x).
        '''
        return self._f

    def get_g(self):
        '''
        Gets the value of g(x).
        '''
        return self._g

    def dynamics(self):
        '''
        General affine system dynamics.
        '''
        self.f()
        self.g()
        self._dstate = self._f + self._g @ self._control

    @abstractmethod
    def f(self):
        pass

    @abstractmethod
    def g(self):
        pass

class Integrator(AffineSystem):
    '''
    Implements a simple n-order integrator.
    '''
    def __init__(self, initial_state, initial_control):
        if len(initial_state) != len(initial_control):
            raise Exception('Number of inputs is different than the number of states.')
        super().__init__(initial_state, initial_control)
        self.f()
        self.g()

    def f(self):
        self._f = np.zeros(self.n)

    def g(self):
        self._g = np.eye(self.n)

class LinearSystem(AffineSystem):
    '''
    Implements a linear system dx = A x + B u.
    '''
    def __init__(self, initial_state, initial_control, A, B):
        super().__init__(initial_state, initial_control)
        self._A = A
        self._B = B

    def f(self):
        self._f = self._A @ self._state

    def g(self):
        self._g = self._B

class Periodic(AffineSystem):
    '''
    Implements a general periodic dynamic: dx = \sum_{k,i} A_{k,i} sin(w_{k,i} x_k + \phi_{k,i})
    '''
    def __init__(self, initial_state, initial_control, **kwargs):
        super().__init__(initial_state, initial_control)
        self.set_param(**kwargs)

    def set_param(self, **kwargs):
        '''
        Sets the system parameters.
        '''
        for key in kwargs:
            if key == "Gains":
                self._gains = kwargs[key]
            if key == "Frequencies":
                self._frequencies = kwargs[key]
            if key == "Phases":
                self._phases = kwargs[key]

        self.sh = self._gains.shape[:2]
        self._num_terms = self.sh[0]
        if self.sh[1] != self.n:
            raise Exception('Input dimensions mismatch.')

        if self.sh != self._frequencies.shape or self.sh != self._phases.shape:
            raise Exception('Input dimensions mismatch.')

        for i in range(self.sh[0]):
            for j in range(self.sh[1]):
                if self._gains[i,j].shape != (self.n, self.m) or self._frequencies.shape != (self.n, self.m) or self._phases.shape != (self.n, self.m):
                    raise Exception('Input dimensions mismatch.')

    def f(self):
        self._f = np.zeros(self.n)

    def g(self):
        self._g = np.zeros([self.n, self.m])
        for i in range(self._num_terms):
            for k in range(self.n):
                self._g += self._gains[k,i] * np.sin( self._frequencies[k,i] * self._state[k] + self._phases[k,i] )

class Unicycle(AffineSystem):
    '''
    Implements the unicycle dynamics: dx = v cos(phi), dy = v sin(phi), dphi = omega.
    State and control are given by [x, y, z] and [v, omega], respectively.
    '''
    from common import Rect
    def __init__(self, initial_state, initial_control, geometric_params=Rect([1.0, 1.0], 0.0)):
        if len(initial_state) != 3:
            raise Exception('State dimension is different from 3.')
        if len(initial_control) != 2:
            raise Exception('Control dimension is different from 2.')
        super().__init__(initial_state, initial_control)
        self.geometry = geometric_params
        self.pos_offset = self.geometry.center_offset
        self.f()
        self.g()

    def f(self):
        self._f = np.zeros(self.n)

    def g(self):
        phi = self._state[2]
        self._g = np.array([[ np.cos(phi), -self.pos_offset*np.sin(phi) ],[ np.sin(phi), self.pos_offset*np.cos(phi) ],[0.0, 1.0]])

class PolynomialSystem(AffineSystem):
    '''
    Class for affine polynomial systems of the type xdot = f(x) + g(x) u, where f(x), g(x) are affine combinations of monomials:
        f(x) = sum F_k m_k(x),
        g(x) = sum G_k m_k(x),
    where m(x) = [ m_1(x) m_2(x) ... m_p(x) ] is a vector of (n+d,d) known monomials (up to degree d) and:
        F = [ F_1 F_2 ... F_p ], F_k in (n x 1)
        G = [ G_1 G_2 ... G_p ], G_k in (n x m)
    are lists of vector and matrix coefficients describing f(x) and g(x).
    '''
    def __init__(self, initial_state, initial_control, **kwargs):
        
        super().__init__(initial_state, initial_control)
        self.set_param(**kwargs)
        
        # self._symbols = sp.symarray('x',self.n)
        # self._monomials = Basis.from_degree(self.n, self._degree).to_sym(self._symbols)
        # self._num_monomials = len(self._monomials)

        import sympy as sp
        self._symbols = []
        for dim in range(self.n):
            self._symbols.append( sp.Symbol('x' + str(dim+1)) )

        from common import generate_monomial_list, generate_monomials_from_symbols
        alpha, _ = generate_monomial_list(self.n, self._degree)
        self._monomials = generate_monomials_from_symbols( self._symbols, alpha )
        self._num_monomials = len(self._monomials)

        # Symbolic f(x) and corresponding lambda function
        self._sym_f = sp.zeros(self.n,1)
        for k in range(len(self._F_list)):
            self._sym_f += sp.Matrix(self._F_list[k]) * self._monomials[k]
        self._lambda_f = sp.lambdify( list(self._symbols), self._sym_f )

        # Symbolic g(x)
        self._sym_g = sp.zeros(self.n,self.m)
        for k in range(len(self._G_list)):
            self._sym_g += sp.Matrix(self._G_list[k]) * self._monomials[k]
        self._lambda_g = sp.lambdify( list(self._symbols), self._sym_g )

    def set_param(self, **kwargs):
        '''
        Sets the system parameters.
        '''
        self._degree = 0
        self._num_monomials = 1

        for key in kwargs:
            if key == "degree":
                self._degree = kwargs[key]
            if key == "F_list":
                self._F_list = kwargs[key]
            if key == "G_list":
                self._G_list = kwargs[key]

        self._num_monomials = scipy.special.comb( self.n+self._degree, self._degree, exact=True )

        # Initialize lists with zeros if no argument is given.
        if not hasattr(self, "_F_list"):
            self._F_list = [ np.zeros(self.n) for _ in range(self._num_monomials) ]
        if not hasattr(self, "_G_list"):
            self._G_list = [ np.zeros([self.n,self.m]) for _ in range(self._num_monomials) ]

        # Debugging
        if len(self._F_list) != self._num_monomials or len(self._G_list) != self._num_monomials:
            raise Exception("Number of list elements must be " + str(self._num_monomials) + ".")

        for F_k in self._F_list:
            if len(F_k) != self.n:
                raise Exception("The dimension of the vector monomial coefficients of f(x) must be n!")

        for G_k in self._G_list:
            if np.shape(G_k) != (self.n, self.m):
                raise Exception("The dimension of the matrix monomial coefficients of g(x) must be (n x m)!")

    def f(self):
        self._f = self._lambda_f(*self._state).reshape(self.n,)

    def g(self):
        self._g = self._lambda_g(*self._state).reshape(self.n,self.m)

    def get_symbols(self):
        '''
        Return the sympy variables.
        '''
        return list(self._symbols)

    def get_monomials(self):
        '''
        Return the monomial basis vector.
        '''
        return self._monomials

    def get_symbolic_f(self):
        '''
        Return symbolic expression for f(x)
        '''
        return self._sym_f

    def get_symbolic_g(self):
        '''
        Return symbolic expression for g(x)
        '''
        return self._sym_g

    def get_F(self):
        '''
        Return polynomial coefficients of f(x)
        '''
        return self._F_list

    def get_G(self):
        '''
        Return polynomial coefficients of g(x)
        '''
        return self._G_list

    def get_monomial_dimension_f(self):
        '''
        Return the monomial dimension of f(x).
        '''
        return len(self._F_list)

    def get_monomial_dimension_g(self):
        '''
        Return the monomial dimension of g(x).
        '''
        return len(self._G_list)
    
class KernelAffineSystem(AffineSystem):
    '''
    Class for affine nonlinear systems of the type xdot = f(x) + g(x) u, where f(x) is given by the gradient
        f(x) = g(x) g'(x) d ( m(x)' F m(x) ) / dx
    where m(x) is a kernel function and F is a suitable matrix of the kernel dimension.
    '''
    def __init__(self, initial_state, initial_control, kernel, F, g_method):

        # Initialize kernel function
        from functions import Kernel
        if type(kernel) != Kernel:
            raise Exception("Argument must be a valid Kernel function.")
        self.kernel = kernel
        self.kernel_dim = self.kernel._num_monomials

        # Initialize F matrix
        if np.shape(F) != (self.kernel_dim, self.kernel_dim):
            raise Exception('F must be a matrix of the same dimension as the kernel.')
        self.F = F

        # Initialize g method
        self.g_method = g_method
      
        super().__init__(initial_state, initial_control)
        self.f()

    def get_g(self, x):
        return self.g_method(x)

    def get_fc(self, x):
        m = self.kernel.function(x)
        Jac = self.kernel.jacobian(x)
        return Jac.T @ self.F @ m

    def get_f(self, x):
        g = self.get_g(x)
        G = g @ g.T
        return G @ self.get_fc(x)

    def get_F(self):
        return self.F

    def f(self):
        self._f = self.get_f(self._state)

    def g(self):
        self._g = self.get_g(self._state)

    def __str__(self):
        text = "Conservative affine nonlinear system of the type dx = f(x) + g(x) u ,\n"
        text += "where f(x) = g(x) g'(x) d( m'(x) F m(x) )/dx, \n"
        text += "with kernel function as \n"
        text += self.kernel.__str__()
        return text

class KernelAffineSystem2(AffineSystem):
    '''
    Class for affine nonlinear systems of the type xdot = f(x) + g(x) u, where:
    - f(x), g(x) are given component-wise by kernel quadratic forms, respectively as fi(x) = m(x)' Fi m(x) and gij(x) = Gij' m(x)
    where m(x) is a kernel function and F is a suitable matrix of the kernel dimension.
    '''
    def __init__(self, initial_state, initial_control, kernel, F, g_method):

        # Initialize kernel function
        from functions import Kernel
        if type(kernel) != Kernel:
            raise Exception("Argument must be a valid Kernel function.")
        self.kernel = kernel
        self.kernel_dim = self.kernel._num_monomials

        # Initialize F matrix
        if np.shape(F) != (self.kernel_dim, self.kernel_dim):
            raise Exception('F must be a matrix of the same dimension as the kernel.')
        self.F = F

        # Initialize g method
        self.g_method = g_method
      
        super().__init__(initial_state, initial_control)
        self.f()

    def get_g(self, x):
        return self.g_method(x)

    def get_fc(self, x):
        m = self.kernel.function(x)
        Jac = self.kernel.jacobian(x)
        return Jac.T @ self.F @ m

    def get_f(self, x):
        g = self.get_g(x)
        G = g @ g.T
        return G @ self.get_fc(x)

    def get_F(self):
        return self.F

    def f(self):
        self._f = self.get_f(self._state)

    def g(self):
        self._g = self.get_g(self._state)

    def __str__(self):
        text = "Conservative affine nonlinear system of the type dx = f(x) + g(x) u ,\n"
        text += "where f(x) = g(x) g'(x) d( m'(x) F m(x) )/dx, \n"
        text += "with kernel function as \n"
        text += self.kernel.__str__()
        return text