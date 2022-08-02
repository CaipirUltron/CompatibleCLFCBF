import numpy as np
import picos as pc
import sympy as sp

from quadratic_program import QuadraticProgram
from functions import PolynomialFunction


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

        self._alpha = PolynomialFunction.generate_monomial_list(self._state_dim, self._lyapunov_degree)

        self._dV_degree = 3*self._lyapunov_degree
        self._dV_alpha = PolynomialFunction.generate_monomial_list(self._state_dim, self._dV_degree)
        self._dV_monomials = PolynomialFunction.generate_monomials_from_symbols(self._lyapunov_symbols, self._dV_degree)

        # PICOS variable
        self._mon_dim = len(self._lyapunov_monomials)
        self._f_dim = self._plant.get_monomial_dimension_f()
        self._g_dim = self._plant.get_monomial_dimension_g()

        self._picos_P = pc.SymmetricVariable("P", self._mon_dim)
        self._picos_u = pc.RealVariable("u", self._ctrl_dim)
        self.compute_lambda_tensor()

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

    def compute_lambda_tensor(self):
        '''
        Computes the tensor = phi(P,u) n(x) (n(x) is a vector of monomials)
        '''
        G_list = self._plant.get_G()
        A_list = self._lyapunov.get_matrices()

        symbolic_P = sp.Matrix(sp.symarray( 'p', (self._mon_dim, self._mon_dim)))
        symbolic_u = sp.symarray( 'u', self._ctrl_dim )

        self.symbolic_M = np.empty((self._mon_dim, self._g_dim, self._mon_dim), dtype=object)
        self.picos_M = np.empty((self._mon_dim, self._g_dim, self._mon_dim), dtype=object)
        for r in range(self._mon_dim):
            for s in range(self._g_dim):
                for t in range(self._mon_dim):
                    for i in range(self._mon_dim):
                        for j in range(self._ctrl_dim):
                            for k in range(self._state_dim):
                                self.symbolic_M[r,s,t] = symbolic_P[r,i]*A_list[k][i,t]*G_list[s][k,j]*symbolic_u[j]
                                self.picos_M[r,s,t] = self._picos_P[r,i]*A_list[k][i,t]*G_list[s][k,j]*self._picos_u[j]

        self.lambda_M =  sp.lambdify( [symbolic_P, symbolic_u], self.symbolic_M )

    def compute_tensor(self, P, u):
        '''
        Computes the numeric value of the M tensor.
        '''
        return self.lambda_M(P.reshape(np.size(P)), u)

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