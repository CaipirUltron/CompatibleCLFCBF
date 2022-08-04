import numpy as np
import picos as pc
import sympy as sp

from quadratic_program import QuadraticProgram
from common import *

class SoSController():
    '''
    Class for the nominal QP controller.
    '''
    def __init__(self, plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 1.0, epsilon = 1.0 , dt = 0.001):

        # Dimensions and system model initialization
        self._plant = plant
        self._lyapunov = clf
        self._barrier = cbf
        self._state_dim = self._plant.n
        self._ctrl_dim = self._plant.m

        # self._state_symbols = self.clf.get_symbols()
        # self._clf_poly = self.clf.get_polynomial()
        # self._cbf_poly = self.cbf.get_polynomial()

        # self.state_dim = self.plant.n
        # self.control_dim = self.plant.m
        # self.sym_dim = int(( self.state_dim * ( self.state_dim + 1 ) )/2)
        # self.skewsym_dim = int(( self.state_dim * ( self.state_dim - 1 ) )/2)
        
        # Important monomials
        self._lyapunov_degree = self._lyapunov.get_degree()
        self._lyapunov_symbols = self._lyapunov.get_symbols()
        self._lyapunov_monomials = self._lyapunov.get_monomials()

        self._alpha = generate_monomial_list(self._state_dim, self._lyapunov_degree)

        self._dV_degree = 3*self._lyapunov_degree
        self._dV_alpha = generate_monomial_list(self._state_dim, self._dV_degree)
        self._dV_monomials = generate_monomials_from_symbols(self._lyapunov_symbols, self._dV_degree)

        # PICOS variable
        self._mon_lyap_dim = len(self._lyapunov_monomials)
        self._mon_plant_dim = self._plant._num_monomials

        self._f_dim = self._plant.get_monomial_dimension_f()
        self._g_dim = self._plant.get_monomial_dimension_g()

        self._picos_P = pc.SymmetricVariable("P", self._mon_lyap_dim)
        self._picos_u = pc.RealVariable("u", self._ctrl_dim)
        self.compute_dot_lyapunov()

        # QP parameters
        self.p = p
        self.gamma = gamma
        self._qp_dim = self._ctrl_dim + 1
        P = np.eye(self._qp_dim)
        P[self._ctrl_dim,self._ctrl_dim] = self.p
        q = np.zeros(self._qp_dim)
        self.QP = QuadraticProgram(P=P, q=q)

        # SoS optimization

        # self.epsilon = epsilon
        # self.pos_def_poly = self.epsilon * PolynomialFunction(np.zeros(self.state_dim), degree = 1, P = np.diag([0.0, 1.0, 1.0,])).get_polynomial()
        
        # self.positivity_constr = self.SoS.add_sos_constraint(self._clf_poly - self.pos_def_poly, self._state_symbols, name="positivity")
        # self.dissipativity_constr = self.SoS.add_sos_constraint(, self.clf.get_symbols(), name="dissipativity")

        self.ctrl_dt = dt

    def compute_bilinear_operator(self):
        '''
        Computes the bilinear operator phi(P,u) such that dV = phi(P,u) n(x) (n(x) is a vector of monomials)
        '''
        self.phi = np.zeros(len(self._dV_monomials))
        pass

    def compute_dot_lyapunov(self):
        '''
        Computes the time derivative of the Lyapunov function along the system trajectories.
        '''
        # Computes P-dependent vector polynomial
        lyap_monomials = np.array(self._lyapunov.get_monomials())
        A_list = self._lyapunov.get_matrices()
        P_sp = sp.Matrix(sp.symarray( 'p', (self._mon_lyap_dim, self._mon_lyap_dim)))

        P_sp_poly = np.empty(self._state_dim, dtype=object)
        P_pc_poly = np.empty(self._state_dim, dtype=object)
        for k in range(self._state_dim):
            P_sp_poly[k] = lyap_monomials.dot( ( P_sp @ A_list[k] ) @ lyap_monomials )
            # P_pc_poly[k] = lyap_monomials.dot( ( self._picos_P @ A_list[k] ) @ lyap_monomials )

        # Computes u-dependent vector polynomial
        plant_monomials = np.array(self._plant.get_monomials())
        G_list = self._plant.get_G()
        u_sp = sp.symarray( 'u', self._ctrl_dim )

        u_sp_poly = 0.0
        u_pc_poly = 0.0
        for k in range(self._mon_plant_dim):
            u_sp_poly += plant_monomials[k]*G_list[k] @ u_sp
            # u_pc_poly += plant_monomials[k]*G_list[k] @ self._picos_u

        # Computes dot product
        self._dot_lyapunov_sp = P_sp_poly.dot( u_sp_poly )
        # self._dot_lyapunov_pc = P_pc_poly.dot( u_pc_poly )

        # self.lambda_dot_lyapunov = sp.lambdify( [P_sp, u_sp], self._dot_lyapunov_sp )

    # def compute_tensor(self, P, u):
    #     '''
    #     Computes the numeric value of the M tensor.
    #     '''
    #     return self.lambda_dot_lyapunov(P.reshape(np.size(P)), u)

    def get_control(self):
        '''
        Computes the QP control.
        '''
        a_clf, b_clf = self.get_clf_constraint()
        a_cbf, b_cbf = self.get_cbf_constraint()

        # Stacking the CLF and CBF constraints
        A = np.vstack( [a_clf, a_cbf ])
        b = np.array([ b_clf, b_cbf ],dtype=float)

        # Solve QP
        self.QP.set_constraints(A, b)
        QP_sol = self.QP.get_solution()
        control = QP_sol[0:self.control_dim,]

        return control

    def get_clf_constraint(self):
        '''
        Sets the Lyapunov constraint.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()
        state = self.plant.get_state()

        # Lyapunov function and gradient
        self.V = self.clf.evaluate_function(*state)[0]
        self.nablaV = self.clf.evaluate_gradient(*state)[0]
        
        # Lie derivatives
        self.LfV = self.nablaV.dot(f)
        self.LgV = g.T.dot(self.nablaV)

        # CLF contraint for the QP
        a_clf = np.hstack( [ self.LgV, -1.0 ])
        b_clf = -self.gamma * self.V - self.LfV

        return a_clf, b_clf

    def get_cbf_constraint(self):
        '''
        Sets the barrier constraint.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()
        state = self.plant.get_state()

        # Barrier function and gradient
        self.h = self.cbf.evaluate_function(*state)[0]
        self.nablah = self.cbf.evaluate_gradient(*state)[0]

        self.Lfh = self.nablah.dot(f)
        self.Lgh = g.T.dot(self.nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack( [ self.Lgh, 0.0 ])
        b_cbf = self.alpha * self.h + self.Lfh

        return a_cbf, b_cbf