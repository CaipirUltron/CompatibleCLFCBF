from compatible_clf_cbf.dynamic_systems.dynamic_systems import QuadraticFunction
import numpy as np
import scipy
from qpsolvers import solve_qp, solve_safer_qp
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier
from compatible_clf_cbf.dynamic_simulation import SimulateDynamics

class QuadraticProgram():

    def __init__(self, **kwargs):

        for key in kwargs:
            if key == 'P':
                P = kwargs[key]
            elif key == 'q':
                q = kwargs[key]
            elif key == 'A':
                A = kwargs[key]
            elif key == 'b':
                b = kwargs[key]

        if 'P' in locals() and 'q' in locals():
            self.set_cost(P,q)

        if 'A' in locals() and 'b' in locals():
            self.set_constraints(A,b)
        else:
            self.A = None
            self.b = None

        self.last_solution = None

    # Set cost of the type x'Px + q'x 
    def set_cost(self, P, q):

        if ( P.ndim != 2 or q.ndim != 1 ):
            raise Exception('P must be a 2-dim array and q must be a 1-dim array.')

        if P.shape[0] != P.shape[1]:
            raise Exception('P must be a square matrix.')

        if P.shape[0] != len(q):
            raise Exception('A and b must have the same number of lines.')

        self.P = P
        self.q = q
        self.dimension = len(q)

    # Set constraints of the type A x <= b
    def set_constraints(self, A, b):

        if A.ndim != 2 or b.ndim != 1:
            raise Exception('A must be a 2-dim array must b must be a 1-dim array.')

        if A.shape[0] != len(b):
            raise Exception('A and b must have the same number of lines.')

        self.A = A
        self.b = b
        self.dimension = A.shape[0]
        self.num_constraints = A.shape[1]

    # Returns the solution of the configured QP.
    def get_solution(self):

        self.solve_QP()
        return self.last_solution

    # Method for solving the configured QP using quadprog.
    def solve_QP(self):

        # try:
            self.last_solution = solve_qp(P=self.P, q=self.q, G=self.A, h=self.b, solver="quadprog")
        # except Exception as error:
        #     print(error)


class QPController():
    
    def __init__(self, plant, clf, cbf, gamma = [1.0, 1.0], alpha = [1.0, 1.0], p = [10.0, 10.0], dt = 0.001):

        # Initialize plant
        self._plant = plant

        # Initialize active CLF
        self.clf = clf
        self.Hv = self.clf.hessian_matrix
        clf_eig, clf_angle, clf_rot = self.clf.compute_eig()
        self.clf_eig = clf_eig
        self.clf_angle = clf_angle
        self.clf_rot = clf_rot
        self.x0 = self.clf.critical_point

        # Initialize reference CLF
        self.ref_clf = clf
        self.ref_Hv = self.ref_clf.hessian_matrix

        # Initialize CBF
        self.cbf = cbf
        self.Hh = self.cbf.hessian_matrix
        self.p0 = self.cbf.critical_point

        # Dimensions and system model initialization
        self.state_dim = self._plant.state_dim
        self.control_dim = self._plant.control_dim

        # Store eigenbasis for Hv
        n = self.state_dim
        self.eigen_basis = np.zeros([n, n, n])
        for k in range(n):
            self.eigen_basis[:][:][k] = np.outer( self.clf_rot[:,k], self.clf_rot[:,k] )

        # Initialize Hessian CLF
        self.Vpi = 0.0
        self.gradVpi = np.zeros(self.state_dim)

        # Initialize compatibility function
        self.compute_compatibility()

        # Parameters for the inner and outer QPs
        self.gamma = gamma
        self.alpha = alpha

        # Parameters for inner QP controller
        self.inner_QP_dim = self.control_dim + 1
        P_inner = np.eye(self.inner_QP_dim)
        P_inner[self.control_dim,self.control_dim] = p[0]
        q_inner = np.zeros(self.inner_QP_dim)
        self.innerQP = QuadraticProgram(P=P_inner, q=q_inner)

        # Parameters for outer QP controller
        self.outer_QP_dim = self.state_dim + 1
        P_outer = np.eye(self.outer_QP_dim)
        P_outer[self.state_dim,self.state_dim] = p[1]
        q_outer = np.zeros(self.outer_QP_dim)
        self.outerQP = QuadraticProgram(P=P_outer, q=q_outer)

        # Initialize control and relaxation variable
        self.control = np.zeros(self.control_dim)
        self.delta = 0.0
        self.deltapi = 0.0

        # Control sample time
        self.ctrl_dt = dt

        # Initialize dynamic subsystems
        f_integrator, g_integrator = list(), list()
        state_string, ctrl_string = str(), str()
        EYE = np.eye(self.state_dim)
        for k in range(self.state_dim):
            f_integrator.append('0')
            g_integrator.append(EYE[k,:])
            state_string = state_string + 'lambda' + str(k+1) + ', '
            ctrl_string = ctrl_string + 'dlambda' + str(k+1) + ', '

        # Integrator sybsystem for the CLF eigenvalues
        eig_integrator = AffineSystem(state_string, ctrl_string, f_integrator, *g_integrator)
        self.clf_dynamics = SimulateDynamics(eig_integrator, self.clf_eig)

        #---------------------------------
        # self.update_clf(Hv = np.diag([ 5.0, 1.0 ]))
        #---------------------------------

    # This function returns the inner QP control
    def compute_control(self, state):

        a_clf, b_clf = self.compute_clf_constraint(state)
        a_cbf, b_cbf = self.compute_cbf_constraint(state)

        # Stacking the CLF and CBF constraints
        A_inner = np.vstack([a_clf, a_cbf])
        b_inner = np.array([b_clf, b_cbf],dtype=float)

        # Solve inner QP
        self.innerQP.set_constraints(A = A_inner,b = b_inner)
        innerQP_sol = self.innerQP.get_solution()

        self.last_control = innerQP_sol[0:self.control_dim,]
        self.last_delta = innerQP_sol[self.control_dim,]

        return self.last_control, self.last_delta

    # This function implements the Lyapunov constraint
    def compute_clf_constraint(self, state):

        # Affine plant dynamics
        f = self._plant.compute_f(state)
        g = self._plant.compute_g(state)

        # Lyapunov function and gradient
        V = self.clf(state)
        nablaV = self.clf.gradient(state)

        LfV = nablaV.dot(f)
        LgV = g.T.dot(nablaV)

        # CLF contraint for the QP
        a_clf = np.hstack( [ LgV, -1.0 ])
        b_clf = -self.gamma[0] * V - LfV

        return a_clf, b_clf

    # This function implements the barrier constraint
    def compute_cbf_constraint(self, state):

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
        b_cbf = self.alpha[0] * h + Lfh

        return a_cbf, b_cbf

    # Updates the active CLF function
    def update_clf(self, **kwargs):

        for key in kwargs:
            if key == 'Hv':
                self.Hv = kwargs[key]
                self.clf.set_hessian(self.Hv)
            elif key == 'x0':
                self.x0 = kwargs[key]
                self.clf.set_critical(self.x0)

        self.compute_compatibility()

    # Updates the active CBF function
    def update_cbf(self, **kwargs):

        for key in kwargs:
            if key == 'Hh':
                self.Hh = kwargs[key]
                self.cbf.set_hessian(self.Hh)
            elif key == 'p0':
                self.p0 = kwargs[key]
                self.cbf.set_critical(self.p0)

        self.compute_compatibility()

    # Receives a control and updates
    def update_clf_dynamics(self, lambdav_ctrl):

        # Integrate CLF eigenvalue subsystem
        self.clf_dynamics.send_control_inputs(lambdav_ctrl, self.ctrl_dt)

        # Get CLF eigenvalue state and compute updated hessian matrix
        n = self.state_dim
        Hv = np.zeros([n,n])
        lambdav = self.clf_dynamics.state()
        for k in range(len(lambdav)):
            Hv = Hv + lambdav[k] * np.outer( self.clf_rot[:,k], self.clf_rot[:,k] )
        
        # Update CLF hessian eigenvalues
        self.update_clf(Hv = Hv)

    # This function returns the outer QP control
    def compute_lambdav_control(self):

        A_outer, b_outer = self.compute_rate_constraint()
        # a_clf_pi, b_clf_pi = self.compute_outer_Lyapunov_constraint()

        # Stacking the CLF and CBF constraints
        # A_outer = np.vstack([a_clf_pi, a_cbf_pi])
        # b_outer = np.array([b_clf_pi, b_cbf_pi],dtype=float)

        # Solve inner QP
        self.outerQP.set_constraints(A = A_outer,b = b_outer)
        outerQP_sol = self.outerQP.get_solution()

        self.last_lambdav_ctrl = outerQP_sol[0:self.state_dim,]
        self.last_deltapi_ctrl = outerQP_sol[self.state_dim,]

        return self.last_lambdav_ctrl, self.last_deltapi_ctrl

    # This function implements the Lyapunov constraint for the outer subsystem
    def compute_rate_constraint(self):

        # Computes the Hessian error (Hv - Hv,ref)
        deltaHv = self.Hv - self.ref_Hv

        # Computes the CLF for the outer subsystem
        self.Vpi = 0.5 * np.trace( np.matmul(deltaHv,deltaHv) )

        # Computes the gradient of the CLF for the outer subsystem
        self.nablaVpi = np.zeros(self.state_dim)
        for k in range(self.state_dim):
            self.nablaVpi[k] = np.trace( np.matmul( deltaHv, self.eigen_basis[:][:][k]) )

        # CLF contraint for the outer QP 
        a_clf_pi = np.hstack( [ self.nablaVpi, -1.0 ])
        b_clf_pi = -self.gamma[1] * self.Vpi

        return a_clf_pi, b_clf_pi

    # This function implements the Lyapunov constraint for the outer subsystem
    def compute_compatibility_constraint(self):
        return

    # This function computes the polynomials of the rational compatibility funcion f(\lambda). It assumes an invertible Hv.
    def compute_compatibility(self):

        n = self.state_dim
                
        # Similarity transformation
        self.v0 = self.Hv.dot( self.p0 - self.x0 )

        # Get the generalized Schur decomposition of the matrix pencil and compute the generalized eigenvalues
        schurHv, schurHh, alpha, beta, Q, Z = scipy.linalg.ordqz(self.Hv, self.Hh)
        for k in range(n):
            self.pencil_char_roots = alpha/beta

        # Assumption: Hv is invertible
        self.detHv = np.linalg.det(self.Hv)
        self.detHh = np.linalg.det(self.Hh)
        try:
            Hv_inv = np.linalg.inv(self.Hv)
            Hv_adj = self.detHv * Hv_inv
        except np.linalg.LinAlgError as error:
            print(error)
            return

        # Computes the pencil characteristic polynomial and denominator of f(\lambda)
        self.pencil_char = ( np.real(np.prod(self.pencil_char_roots))/self.detHv ) * np.real(np.polynomial.polynomial.polyfromroots(self.pencil_char_roots))
        self.den_poly = np.polymul(self.pencil_char, self.pencil_char)

        # This computes the pencil adjugate expansion and the set of numerator vectors by the adapted Faddeev-LeVerrier algorithm.
        D = np.zeros([n, n, n])
        D[:][:][0] = pow(-1,n-1) * Hv_adj

        Omega = np.zeros([n,n])
        Omega[0,:] = D[:][:][0].dot(self.v0)

        for k in range(1,n):
                D[:][:][k] = Hv_inv * ( self.Hh * D[:][:][k-1] - self.pencil_char[k]*np.eye(n) )
                Omega[k,:] = D[:][:][k].dot(self.v0)        

        # Computes the numerator polynomial
        W = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                W[i,j] = self.Hh.dot(Omega[i,:]).dot(Omega[j,:])

        EYE = np.eye(n)
        self.num_poly = np.zeros(n+1)
        for k in range(n):
            poly_term = np.polymul( W[:,k], EYE[k,:] )
            self.num_poly = np.polyadd(self.num_poly, poly_term)

        # Computes polynomial roots
        self.num_roots = np.polynomial.polynomial.polyroots(self.num_poly)
        self.den_roots = np.polynomial.polynomial.polyroots(self.den_poly)

        # Computes f(0)
        self.f0 = self.num_poly[0]/pow(self.pencil_char[0],2)