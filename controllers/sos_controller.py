import numpy as np
from quadratic_program import QuadraticProgram
from functions import PolynomialFunction
from SumOfSquares import SOSProblem, poly_opt_prob, Basis


class SoSController():
    '''
    Class for the nominal QP controller.
    '''
    def __init__(self, plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 1.0, epsilon = 1.0 , dt = 0.001):

        # Dimensions and system model initialization
        self.plant = plant
        self.clf = clf
        self.cbf = cbf

        self._state_symbols = self.clf.get_symbols()
        self._clf_poly = self.clf.get_polynomial()
        self._cbf_poly = self.cbf.get_polynomial()

        self.state_dim = self.plant.n
        self.control_dim = self.plant.m
        self.sym_dim = int(( self.state_dim * ( self.state_dim + 1 ) )/2)
        self.skewsym_dim = int(( self.state_dim * ( self.state_dim - 1 ) )/2)
        
        # QP parameters
        self.p = p
        self.gamma, self.alpha = gamma, alpha
        self.QP_dim = self.control_dim + 1
        P = np.eye(self.QP_dim)
        P[self.control_dim,self.control_dim] = self.p
        q = np.zeros(self.QP_dim)
        self.QP = QuadraticProgram(P=P, q=q)

        # SoS optimization
        self.SoS = SOSProblem()

        self.epsilon = epsilon
        self.pos_def_poly = self.epsilon * PolynomialFunction(np.zeros(self.state_dim), degree = 1, P = np.diag([0.0, 1.0, 1.0,])).get_polynomial()
        
        self.positivity_constr = self.SoS.add_sos_constraint(self._clf_poly - self.pos_def_poly, self._state_symbols, name="positivity")
        # self.dissipativity_constr = self.SoS.add_sos_constraint(, self.clf.get_symbols(), name="dissipativity")

        self.ctrl_dt = dt

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

    def update_clf_dynamics(self, piv_ctrl):
        '''
        Integrates the dynamic system for the CLF Hessian matrix.
        '''
        self.clf.update(piv_ctrl, self.ctrl_dt)

    def update_cbf_dynamics(self, pih_ctrl):
        '''
        Integrates the dynamic system for the CBF Hessian matrix.
        '''
        self.cbf.update(pih_ctrl, self.ctrl_dt)

    def get_lambda(self):
        '''
        Computes the KKT multipliers of the Optimization problem.
        '''
        LgV2 = self.LgV.dot(self.LgV)
        Lgh2 = self.Lgh.dot(self.Lgh)
        LgVLgh = self.LgV.dot(self.Lgh)

        delta = LgVLgh**2 - ( (1/self.p) + LgV2 ) * Lgh2
        
        FV = self.LfV + self.gamma * self.V
        Fh = self.Lfh + self.alpha * self.h
        
        lambda1 = (1/delta) * ( Fh * LgVLgh - FV * Lgh2 )
        lambda2 = (1/delta) * ( Fh * ( (1/self.p) + LgV2 ) - FV * LgVLgh )

        return lambda1, lambda2, delta