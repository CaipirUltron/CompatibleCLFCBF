import numpy as np
from cbf_vision.basics import ControlInput
from cbf_vision.vision import DynamicFeature2D
from scipy.optimize import linprog
from qpsolvers import solve_qp, solve_safer_qp

class CBFSinglePoint:

    def __init__(self, gamma = 1.0, alpha = 1.0, epsilon = 0.0, p = 100.0):

        # Dimensions and system model initialization
        self.state_dimension = 6
        self.control_dimension = 6
        self.QP_dimension = self.control_dimension + 1

        # Parameters for QP-based controller
        self.gamma = gamma
        self.alpha = alpha
        self.cost_function_gain = np.eye(self.QP_dimension)
        self.cost_function_gain[self.control_dimension,self.control_dimension] = p
        self.epsilon = epsilon                           # threshold for third CBF
        self.delta = 0

        # Initialize Jacobian matrices
        self.Jv = np.zeros((2,3))
        self.Jw = np.zeros((2,3))
        self.Jq = np.array([[0., 0., 1.]])
        self.Jl = np.zeros((1,3))

    def compute_control_input(self, state, v_ref):
        s = state.feature_coordinates
        chi = state.inverse_depth
        # v = state.linear_velocity
        # w = state.angular_velocity

        a_clf, b_clf = self.compute_Lyapunov_constraints(s, chi)
        a1, b1 = self.computeFirstCondition(s, chi)
        a2, b2 = self.computeSecondCondition(s, chi)
        a_eq, b_eq = self.computeVelocityCondition(s, chi, v_ref)

        # Compute Quadratic Program
        QP_sol = self.compute_QP_solution(a_clf, a1, a2, a_eq, b_clf, b1, b2, b_eq)

        u = QP_sol[0:self.control_dimension,]
        self.delta = QP_sol[self.control_dimension,]

        return ControlInput(linear_velocity=u[0:3], angular_velocity=u[3:6])

    def compute_Lyapunov_constraints(self, s, chi):

        x, y = s[0], s[1]
        state = np.concatenate( (s, chi), axis=0 )

        # Updates Jacobian matrices
        Jv, Jw, Jq, Jl = DynamicFeature2D.compute_matrices(x, y)

        # Lyapunov function and gradient
        V = 0.5*np.linalg.norm(s) ** 2

        a_clf = np.hstack( [ chi*s@Jv, s@Jw, -1.0])
        b_clf = -self.gamma * V

        return a_clf, b_clf

    def computeFirstCondition(self, s, chi):
        x, y = s[0], s[1]

        # Updates Jacobian matrices
        Jv, Jw, Jq, Jl = DynamicFeature2D.compute_matrices(x, y)

        a1 = np.hstack([ np.zeros((1,3)), self.Jl, np.zeros((1,1)) ])
        b1 = 0.0

        return a1, b1

    def computeSecondCondition(self, s, chi):
        x, y = s[0], s[1]

        # Updates Jacobian matrices
        Jv, Jw, Jq, Jl = DynamicFeature2D.compute_matrices(x, y)

        a2 = np.hstack([ self.Jq, np.zeros((1,3)), np.zeros((1,1)) ])
        b2 = 0.0

        return a2, b2

    def computeVelocityCondition(self, s, chi, v_ref):
        x, y = s[0], s[1]

        a_eq = np.hstack([ np.eye(3), np.zeros((3,3)), np.zeros((3,1)) ])
        b_eq = v_ref

        return a_eq, b_eq

    # This function implements the constraints of Theorem 1 as CBFs (from 'Active Depth Estimation: Stability Analysis and its Applications'):
    def compute_barrier_constraints(self, s, chi):
        x, y = s[0], s[1]

        # Updates Jacobian matrices
        Jv, Jw, Jq, Jl = DynamicFeature2D.compute_matrices(x, y)

        ## Third CBF
        # deltax, deltay = (x*vz - vx), (y*vz - vy)
        # sigma2 = pow(deltax,2) + pow(deltay,2)
        # h3 = 0.5*(sigma2 - self.epsilon)
        # Lf_h3 = ( dx*deltax + dy*deltay )*vz
        # Lg_h3 = np.array([[ -deltax, -deltay, ( x*deltax + y*deltay ) ]])

        # CBF constraints
        a1 = np.hstack([ np.zeros((1,3)), self.Jl, np.zeros((1,1)) ])
        b1 = 0.0

        a2 = np.hstack([ self.Jq, np.zeros((1,3)), np.zeros((1,1)) ])
        b2 = 0.0

        # a_cbf3 = -np.hstack([ Lg_h3,np.zeros((1,3+1)) ])
        # b_cbf3 = Lf_h3 + self.alpha * h3

        a4 = np.hstack([ np.eye(3), np.zeros((3,3)), np.zeros((3,1)) ])
        b4 = np.array((.1,0,0))

        a_cbf = np.vstack([a1, a2])
        b_cbf = np.array([b1, b2])

        return a_cbf, b_cbf, a4, b4

    def compute_QP_solution(self, a_clf, a1, a2, a_eq, b_clf, b1, b2, b_eq):
        
        # Stacking the CLF and CBF constraints
        a = np.vstack([a_clf, a1, a2])
        b = np.hstack([b_clf, b1, b2])

        # Solve Quadratic Program of the type: min 1/2x'Hx s.t. A x <= b with quadprog
        # QP_solution = solve_qp(P=self.cost_function_gain, q=np.zeros(self.QP_dimension), G=a, h=b, A=a_eq, b=b_eq, solver="quadprog")
        QP_solution = solve_qp(P=self.cost_function_gain, q=np.zeros(self.QP_dimension), G=a, h=b, solver="quadprog")
        return QP_solution

        # linprog_solution = linprog(c=np.zeros(self.QP_dimension), A_ub=a, b_ub=b, A_eq=a_eq, b_eq=b_eq, solver="quadprog")
        # return linprog_solution

    def compute_QP_test(self, dim, A, b):

            C = np.eye(dim)

            # Solve Quadratic Program of the type: min 1/2x'Hx s.t. A x <= b with quadprog
            QP_solution = solve_qp(C, np.zeros(dim), A, b, solver="quadprog")

            return QP_solution
