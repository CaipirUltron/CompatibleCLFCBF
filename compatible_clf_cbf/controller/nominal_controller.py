import numpy as np
from compatible_clf_cbf.quadratic_program import QuadraticProgram

class NominalQP():
    '''
    Class for the nominal QP controller.
    '''
    def __init__(self, plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0, dt = 0.001):

        # Dimensions and system model initialization
        self._plant = plant
        self.clf, self.cbf = clf, cbf

        self.state_dim = self._plant.state_dim
        self.control_dim = self._plant.control_dim

        # QP parameters
        self.gamma, self.alpha = gamma, alpha
        self.QP_dim = self.control_dim + 1
        P = np.eye(self.QP_dim)
        P[self.control_dim,self.control_dim] = p
        q = np.zeros(self.QP_dim)
        self.QP = QuadraticProgram(P=P, q=q)

    def compute_control(self, state):
        '''
        Computes the QP control.
        '''
        a_clf, b_clf = self.compute_clf_constraint(state)
        a_cbf, b_cbf = self.compute_cbf_constraint(state)

        # Stacking the CLF and CBF constraints
        A = np.vstack( [a_clf, a_cbf ])
        b = np.array([ b_clf, b_cbf ],dtype=float)

        # Solve QP
        self.QP.set_constraints(A, b)
        QP_sol = self.QP.get_solution()
        control = QP_sol[0:self.control_dim,]

        return control

    def compute_clf_constraint(self, state):
        '''
        Sets the Lyapunov constraint.
        '''
        # Affine plant dynamics
        f = self._plant.compute_f(state)
        g = self._plant.compute_g(state)

        # Lyapunov function and gradient
        V = self.clf(state)
        nablaV = self.clf.gradient(state)

        # Lie derivatives
        LfV = nablaV.dot(f)
        LgV = g.T.dot(nablaV)

        # CLF contraint for the QP
        a_clf = np.hstack( [ LgV, -1.0 ])
        b_clf = -self.gamma * V - LfV

        return a_clf, b_clf

    def compute_cbf_constraint(self, state):
        '''
        Sets the barrier constraint.
        '''
        # Affine plant dynamics
        f = self._plant.compute_f(state)
        g = self._plant.compute_g(state)

        # Barrier function and gradient
        h = self.cbf(state)
        nablah = self.cbf.gradient(state)

        Lfh = nablah.dot(f)
        Lgh = g.T.dot(nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack( [ Lgh, 0.0 ])
        b_cbf = self.alpha * h + Lfh

        return a_cbf, b_cbf