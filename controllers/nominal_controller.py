import numpy as np
from controllers import CLFCBFPair
from quadratic_program import QuadraticProgram


class NominalQP():
    '''
    Class for the nominal QP controller.
    '''
    def __init__(self, plant, clf, cbfs, gamma = 1.0, alpha = 1.0, p = 10.0, dt = 0.001):

        # Dimensions and system model initialization
        self.plant = plant
        self.clf, self.cbfs = clf, cbfs
        self.clf_cbf_pairs = []
        self.mode_log = None

        self.state_dim = self.plant.n
        self.control_dim = self.plant.m
        self.sym_dim = int(( self.state_dim * ( self.state_dim + 1 ) )/2)
        self.skewsym_dim = int(( self.state_dim * ( self.state_dim - 1 ) )/2)
        
        # Compute equilibrium points
        self.equilibrium_points = np.zeros([0,self.state_dim])
        for cbf in cbfs:
            clf_cbf_pair = CLFCBFPair(self.clf, cbf)
            self.clf_cbf_pairs.append( clf_cbf_pair )
            self.equilibrium_points = np.vstack([ self.equilibrium_points, clf_cbf_pair.equilibrium_points.T ])

        # QP parameters
        self.p = p
        self.gamma, self.alpha = gamma, alpha
        self.QP_dim = self.control_dim + 1
        P = np.eye(self.QP_dim)
        P[self.control_dim,self.control_dim] = self.p
        q = np.zeros(self.QP_dim)
        self.QP = QuadraticProgram(P=P, q=q)

        self.ctrl_dt = dt

    def get_control(self):
        '''
        Computes the QP control.
        '''
        A, b = self.get_clf_constraint()

        # Stacking the CLF and CBF constraints
        for cbf in self.cbfs:
            a_cbf, b_cbf = self.get_cbf_constraint(cbf)
            A = np.vstack( [ A, a_cbf ])
            b = np.hstack( [ b, b_cbf ])

        # Solve QP
        self.QP.set_inequality_constraints(A, b)
        QP_sol = self.QP.get_solution()
        control = QP_sol[0:self.control_dim,]

        return control

    def get_clf_control(self):
        '''
        This controller will not modify the CLF.
        '''
        return np.zeros(self.sym_dim)

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

    def get_cbf_constraint(self, cbf):
        '''
        Sets the i-th barrier constraint.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()
        state = self.plant.get_state()

        # Barrier function and gradient
        h = cbf.evaluate_function(*state)[0]
        nablah = cbf.evaluate_gradient(*state)[0]

        self.Lfh = nablah.dot(f)
        self.Lgh = g.T.dot(nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack( [ self.Lgh, 0.0 ])
        b_cbf = self.alpha * h + self.Lfh

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

"""  def get_lambda(self):
        '''
        Computes the KKT multipliers of the Optimization problem.
        '''
        LgV2 = self.LgV.dot(self.LgV)
        Lgh2 = self.Lgh.dot(self.Lgh)
        LgVLgh = self.LgV.dot(self.Lgh)
        
        delta = ( (1/self.p) + LgV2 ) * Lgh2 - LgVLgh**2
        
        FV = self.LfV + self.gamma * self.V
        Fh = self.Lfh + self.alpha * self.h
        
        lambda1 = (1/delta) * ( FV * Lgh2 - Fh * LgVLgh )
        lambda2 = (1/delta) * ( FV * LgVLgh - Fh * ( (1/self.p) + LgV2 )  )

        return lambda1, lambda2, delta """