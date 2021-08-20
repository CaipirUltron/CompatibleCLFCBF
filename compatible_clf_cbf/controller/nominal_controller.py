import numpy as np
from compatible_clf_cbf.quadratic_program import QuadraticProgram

class NominalQP():
    '''
    Class for the nominal QP controller.
    '''
    def __init__(self, plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0, dt = 0.001):

        # Dimensions and system model initialization
        self.plant = plant
        self.clf, self.cbf = clf, cbf

        self.state_dim = self.plant.n
        self.control_dim = self.plant.m

        # QP parameters
        self.p = p
        self.gamma, self.alpha = gamma, alpha
        self.QP_dim = self.control_dim + 1
        P = np.eye(self.QP_dim)
        P[self.control_dim,self.control_dim] = self.p
        q = np.zeros(self.QP_dim)
        self.QP = QuadraticProgram(P=P, q=q)

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
        self.V = self.clf(state)
        self.nablaV = self.clf.gradient(state)

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
        self.h = self.cbf(state)
        self.nablah = self.cbf.gradient(state)

        self.Lfh = self.nablah.dot(f)
        self.Lgh = g.T.dot(self.nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack( [ self.Lgh, 0.0 ])
        b_cbf = self.alpha * self.h + self.Lfh

        return a_cbf, b_cbf

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