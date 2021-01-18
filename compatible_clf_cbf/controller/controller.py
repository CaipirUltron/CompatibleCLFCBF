import numpy as np
from scipy.optimize import linprog
from qpsolvers import solve_qp, solve_safer_qp
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier

class QPController():
    
    def __init__(self, plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 100.0):

        self._plant = plant
        self._clf, self._cbf = clf, cbf

        # Dimensions and system model initialization
        self.state_dim = self._plant.state_dim
        self.control_dim = self._plant.control_dim
        self.QP_dimension = self.control_dim + 1

        # Parameters for QP-based controller
        self.gamma, self.alpha = gamma, alpha
        self.cost_function_gain = np.eye(self.QP_dimension)
        self.cost_function_gain[self.control_dim,self.control_dim] = p

        # Initialize control and relaxation variable
        self.control = np.zeros(self.control_dim)
        self.delta = 0

    def compute_control(self, state):
        a_clf, b_clf = self.compute_Lyapunov_constraints(state)
        a_cbf, b_cbf = self.compute_barrier_constraints(state)

        # Compute Quadratic Program
        QP_sol = self.compute_QP_solution(a_clf, b_clf, a_cbf, b_cbf)

        self.control = QP_sol[0:self.control_dim,]
        self.delta = QP_sol[self.control_dim,]

        return self.control

    # This function implements the Lyapunov constraint
    def compute_Lyapunov_constraints(self, state):

        # Affine plant dynamics
        f = self._plant.compute_f(state)
        g = self._plant.compute_g(state)

        # Lyapunov function and gradient
        V = self._clf(state)
        nablaV = self._clf.gradient(state)

        LfV = nablaV.dot(f)
        LgV = g.T.dot(nablaV)

        a_clf = np.hstack( [ LgV, -1.0 ])
        b_clf = -self.gamma * V - LfV

        return a_clf, b_clf

    # This function implements the barrier constraint
    def compute_barrier_constraints(self, state):

        # Affine plant dynamics
        f = self._plant.compute_f(state)
        g = self._plant.compute_g(state)

        # Barrier function and gradient
        h = self._cbf(state)
        nablah = self._cbf.gradient(state)

        Lfh = nablah.dot(f)
        Lgh = g.T.dot(nablah)

        a_cbf = -np.hstack( [ Lgh, 0.0 ])
        b_cbf = self.alpha * h + Lfh

        return a_cbf, b_cbf

    def compute_QP_solution(self, a_clf, b_clf, a_cbf, b_cbf):
        
        # Stacking the CLF and CBF constraints
        a = np.vstack([a_clf, a_cbf])
        b = np.array([b_clf, b_cbf],dtype=float)

        # Solve Quadratic Program of the type: min 1/2x'Hx s.t. A x <= b with quadprog
        QP_solution = solve_qp(P=self.cost_function_gain, q=np.zeros(self.QP_dimension), G=a, h=b, solver="quadprog")

        return QP_solution