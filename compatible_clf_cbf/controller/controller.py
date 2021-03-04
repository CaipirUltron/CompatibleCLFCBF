import numpy as np
from scipy.optimize import linprog
from qpsolvers import solve_qp, solve_safer_qp
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier

class QPController():
    
    def __init__(self, plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 100.0):

        # Initialize plant
        self._plant = plant

        # Initialize active CLF
        self._clf = clf
        self.Hv = self._clf.hessian_matrix
        self.x0 = self._clf.critical_point

        # Initialize active CBF
        self._cbf = cbf
        self.Hh = self._cbf.hessian_matrix
        self.p0 = self._cbf.critical_point

        # Dimensions and system model initialization
        self.state_dim = self._plant.state_dim
        self.control_dim = self._plant.control_dim
        self.QP_dimension = self.control_dim + 1

        # Compute v0 and characteristic polynomial
        self.v0 = self.Hv.dot( self.p0 - self.x0 )
        self.computePolynomials()

        # Parameters for QP-based controller
        self.gamma, self.alpha = gamma, alpha
        self.cost_function_gain = np.eye(self.QP_dimension)
        self.cost_function_gain[self.control_dim,self.control_dim] = p

        # Initialize control and relaxation variable
        self.control = np.zeros(self.control_dim)
        self.delta = 0

        # Initialize CLF dynamical system
        f_CLF_dynamics, g_CLF_dynamics = list(), list()
        CLF_state_string, CLF_ctrl_string = str(), str()
        EYE = np.eye(self.state_dim)
        for k in range(self.state_dim):
            f_CLF_dynamics.append('0')
            g_CLF_dynamics.append(EYE[k,:])
            CLF_state_string = CLF_state_string + 'lambdav' + str(k+1) + ', '
            CLF_ctrl_string = CLF_ctrl_string + 'dlambdav' + str(k+1) + ', '
            
        self.clf_dynamics = AffineSystem(CLF_state_string, CLF_ctrl_string, f_CLF_dynamics, *g_CLF_dynamics)

    # This function returns the QP-based control
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

    # This function solves the actual QP
    def compute_QP_solution(self, a_clf, b_clf, a_cbf, b_cbf):
        
        # Stacking the CLF and CBF constraints
        a = np.vstack([a_clf, a_cbf])
        b = np.array([b_clf, b_cbf],dtype=float)

        # Solve Quadratic Program of the type: min 1/2x'Hx s.t. A x <= b with quadprog
        QP_solution = solve_qp(P=self.cost_function_gain, q=np.zeros(self.QP_dimension), G=a, h=b, solver="quadprog")

        return QP_solution

    # This function returns TRUE if the CLF-CBF pair is compatible; and FALSE otherwise.
    def isCompatible(self):

        compatible = True

        return compatible

    # This function computes the coefficients of the matrix pencil characteristic polynomial and f numerator using
    # the Faddeev-LeVerrier algorithm. It assumes an invertible Hv.
    def computePolynomials(self):

        # Initialize Faddeev-LeVerrier algorithm
        n = self.state_dim
        D = np.zeros([n, n, n])
        Omega = np.zeros([n,n])
        W = np.zeros([n,n])
        self.pencil_char = np.zeros(n+1)
        
        detHv = np.linalg.det(self.Hv)
        try:
            Hv_inv = np.linalg.inv(self.Hv)
            Hv_adj = detHv * Hv_inv
        except np.linalg.LinAlgError as error:
            print(error)
            return

        # Main loop of the adapted Faddeev-LeVerrier algorithm for linear matrix pencils.
        # This computes the pencil characteristic polynomial, pencil adjugate expansion and
        # the set of numerator vectors.
        self.pencil_char[0] = pow(-1,n) * detHv
        D[:][:][0] = pow(-1,n-1) * Hv_adj
        Omega[0,:] = D[:][:][0].dot(self.v0)
        for k in range(1,n+1):
            self.pencil_char[k] = (1/k)*np.trace( self.Hh * D[:][:][k-1] )
            if k < n:
                D[:][:][k] = Hv_inv * ( self.Hh * D[:][:][k-1] - self.pencil_char[k]*np.eye(n) )
                Omega[k,:] = D[:][:][k].dot(self.v0)

        # Pencil characteristic polynomial and denominator of f(\lambda) function
        self.den_poly = np.polymul(self.pencil_char, self.pencil_char)

        # Computes the W matrix, for the numerator polynomial
        for i in range(n):
            for j in range(n):
                W[i,j] = self.Hh.dot(Omega[i,:]).dot(Omega[j,:])

        # Computes the numerator polynomial
        EYE = np.eye(n)
        self.num_poly = np.zeros(n+1)
        for k in range(n):
            poly_term = np.polymul( W[:,k], EYE[k,:] )
            self.num_poly = np.polyadd(self.num_poly, poly_term)

        # Computes polynomial roots
        self.pencil_char_roots = np.polynomial.polynomial.polyroots(self.pencil_char)
        self.num_roots = np.polynomial.polynomial.polyroots(self.num_poly)
        self.den_roots = np.polynomial.polynomial.polyroots(self.den_poly)

        # Computes f(0)
        self.f0 = self.num_poly[0]/pow(self.pencil_char[0],2)

        # Computes the numerator and denominator second derivatives for degenerate cases (L'Hopital's rule) 
        # dpencil_char = np.polyder(self.pencil_char)
        # ddpencil_char = np.polyder(dpencil_char)
        # self.LHopital_num = np.polyder(np.polyder(self.num_poly))
        # self.LHopital_den = 2*np.polyadd( np.polymul(dpencil_char,dpencil_char) , np.polymul(self.pencil_char,ddpencil_char) )