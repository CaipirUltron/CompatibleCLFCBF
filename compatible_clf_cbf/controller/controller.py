from compatible_clf_cbf.dynamic_systems.dynamic_systems import QuadraticFunction
import numpy as np
import math, scipy
import copy as cp
from qpsolvers import solve_qp, solve_safer_qp
from compatible_clf_cbf.dynamic_systems import AffineSystem
from compatible_clf_cbf.dynamic_simulation import SimulateDynamics


class QPController():
    '''
    Class for the compatible QP controller.
    '''
    def __init__(self, plant, clf, ref_clf, cbf, gamma = [1.0, 1.0], alpha = [1.0, 1.0], p = [10.0, 10.0], dt = 0.001):

        # Dimensions and system model initialization
        self._plant = plant
        self.clf, self.ref_clf, self.cbf = clf, ref_clf, cbf

        self.state_dim = self._plant.state_dim
        self.control_dim = self._plant.control_dim
        self.symmetric_dim = int(( self.state_dim * ( self.state_dim + 1 ) )/2)
        self.sym_basis = QuadraticFunction.symmetric_basis(self.state_dim)

        # Initialize rate CLF
        self.Vpi = 0.0
        self.gradient_Vpi = np.zeros(self.symmetric_dim)

        # Initialize compatibility function
        self.compute_compatibility()

        # Parameters for the inner and outer QPs
        self.gamma, self.alpha = gamma, alpha

        # Parameters for inner QP controller
        self.inner_QP_dim = self.control_dim + 1
        P_inner = np.eye(self.control_dim + 1)
        P_inner[self.control_dim,self.control_dim] = p[0]
        q_inner = np.zeros(self.control_dim + 1)
        self.innerQP = QuadraticProgram(P=P_inner, q=q_inner)

        # Parameters for outer QP controller
        self.outer_QP_dim = self.symmetric_dim + 1
        P_outer = np.eye(self.outer_QP_dim)
        P_outer[self.symmetric_dim,self.symmetric_dim] = p[1]
        q_outer = np.zeros(self.outer_QP_dim)
        self.outerQP = QuadraticProgram(P=P_outer, q=q_outer)

        # Control sample time
        self.ctrl_dt = dt

        # Initialize dynamic subsystems
        f_integrator, g_integrator = list(), list()
        state_string, ctrl_string = str(), str()
        EYE = np.eye(self.symmetric_dim)
        for k in range(self.symmetric_dim):
            f_integrator.append('0')
            g_integrator.append(EYE[k,:])
            state_string = state_string + 'pi' + str(k+1) + ', '
            ctrl_string = ctrl_string + 'dpi' + str(k+1) + ', '

        # Integrator sybsystem for the CLF parameters
        piv_integrator = AffineSystem(state_string, ctrl_string, f_integrator, *g_integrator)
        piv_init = QuadraticFunction.sym2vector(self.clf.hessian())
        self.clf_dynamics = SimulateDynamics(piv_integrator, piv_init)

    def collinear_norm(self, state):
        '''
        Method that returns the positive collinearity norm.
        '''
        nablaV = self.ref_clf.gradient(state)
        nablah = self.cbf.gradient(state)

        inner_gradients = np.inner( nablaV, nablah )
        if inner_gradients < 0:
            D_parallel = 1/(scipy.linalg.norm(nablaV)**2)
        else:
            Projection = np.eye(self.state_dim) - np.outer( nablah, nablah )/(scipy.linalg.norm(nablah)**2)
            D_parallel = scipy.linalg.norm(Projection.dot(nablaV))/(scipy.linalg.norm(nablaV)**3)
        return D_parallel

    def distance_Pplus(self, state):
        '''
        Method that returns the distance (physical) to the positive collinearity set.
        '''
        x, y = state[0], state[1]
        p0 = self.cbf.critical()
        x_cbf, y_cbf = p0[0], p0[1]
        if y_cbf <= y:
            D_col = abs( x - x_cbf )
        else:
            D_col = scipy.linalg.norm( state - p0 )
        return D_col

    def compute_control(self, state):
        '''
        Computes the inner QP control.
        '''
        self.compute_pi_control(state)

        a_clf, b_clf = self.compute_clf_constraint(state)
        a_cbf, b_cbf = self.compute_cbf_constraint(state)

        # Stacking the CLF and CBF constraints
        A_inner = np.vstack([a_clf, a_cbf])
        b_inner = np.array([b_clf, b_cbf],dtype=float)

        # Solve inner QP
        self.innerQP.set_constraints(A = A_inner,b = b_inner)
        innerQP_sol = self.innerQP.get_solution()
        control = innerQP_sol[0:self.control_dim,]

        return control

    def compute_clf_constraint(self, state):
        '''
        Sets the Lyapunov constraint for the inner loop controller.
        '''
        D_col = self.distance_Pplus(state)

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
        # b_clf = -self.gamma[0] * V - LfV

        rate = self.gamma[0] * SimulateDynamics.sat( D_col, 1.0 )
        b_clf = - rate * V - LfV

        # print("Dist. to col = " + str(D_col))
        # print("Rate = " + str(rate))

        return a_clf, b_clf

    def compute_cbf_constraint(self, state):
        '''
        Sets the barrier constraint for the inner loop controller.
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
        b_cbf = self.alpha[0] * h + Lfh

        return a_cbf, b_cbf

    def update_clf_dynamics(self, piv_ctrl):
        '''
        Integrates the dynamic system for the CLF Hessian matrix.
        '''
        self.clf_dynamics.send_control_inputs(piv_ctrl, self.ctrl_dt)
        pi_v = self.clf_dynamics.state()
        Hv = QuadraticFunction.vector2sym(pi_v)

        self.clf.set_param(hessian = Hv)
        self.compute_compatibility()

    def compute_pi_control(self, state):
        '''
        Computes the outer loop control.
        '''
        a_clf_pi, b_clf_pi = self.compute_rate_constraint(state)
        a_cbf_pi, b_cbf_pi = self.compute_compatibility_constraints(state)

        A_outer = np.vstack([a_clf_pi, a_cbf_pi])
        b_outer = np.array([b_clf_pi, b_cbf_pi],dtype=float)

        # A_outer = a_clf_pi
        # b_outer = np.array([b_clf_pi],dtype=float)

        # A_outer = a_cbf_pi
        # b_outer = b_cbf_pi

        self.outerQP.set_constraints(A = A_outer,b = b_outer)
        outerQP_sol = self.outerQP.get_solution()

        piv_control = outerQP_sol[0:self.symmetric_dim,]

        ##################################### Turn off outer loop controller #####################################
        # piv_control = np.zeros(self.symmetric_dim)

        self.update_clf_dynamics(piv_control)

    def compute_rate_constraint(self, state):
        '''
        Sets the Lyapunov constraint for the outer loop controller.
        '''
        deltaHv = self.clf.hessian() - self.ref_clf.hessian()
        self.Vpi = 0.5 * np.trace( np.matmul(deltaHv, deltaHv) )

        # print("Rate CLF = " + str(self.Vpi))

        for k in range(self.symmetric_dim):
            self.gradient_Vpi[k] = np.trace( np.matmul( deltaHv, self.sym_basis[k]) )

        a_clf_pi = np.hstack( [ self.gradient_Vpi, -1.0 ])
        b_clf_pi = -self.gamma[1] * self.Vpi

        return a_clf_pi, b_clf_pi

    def compute_compatibility_constraints(self, state):
        '''
        Sets the barrier constraints for the outer loop controller, ensuring compatibility.
        '''
        col_norm = self.collinear_norm(state)

        nablaV, nablah = self.clf.gradient(state), self.cbf.gradient(state)
        inner_gradients = np.inner( nablaV, nablah )/(scipy.linalg.norm(nablah)**2)

        gamma_constraint = 10.0
        h_gamma = np.zeros(self.number_critical)
        gradient_h_gamma = np.zeros([self.number_critical, self.symmetric_dim])
        for k in range(self.number_critical):
            h_gamma[k] = np.log( self.critical_values ) - gamma_constraint

            v = self.compute_v_function( self.critical_points[k] )
            H = self.compute_pencil( self.critical_points[k] )
            H_inv = np.linalg.inv(H)
            Hprod = np.matmul( self.cbf.hessian(), H_inv )
            vec_nabla_f = 2 * (1/self.critical_values[k]) * Hprod.dot(v)
            for i in range(self.symmetric_dim):
                vec_i = self.sym_basis[i].dot( v + self.cbf.critical() - self.clf.critical() )
                gradient_h_gamma[k,i] = vec_nabla_f.dot(vec_i)

        a_cbf_pi = -np.hstack([ gradient_h_gamma, np.zeros([self.number_critical,1]) ])
        b_cbf_pi = h_gamma + col_norm * np.ones(self.number_critical)
        # b_cbf_pi = h_gamma

        print("Term = " + str(col_norm))

        return a_cbf_pi, b_cbf_pi

    def compute_v_function(self, lambda_var):
        '''
        This function returns the value of vector v(lambda) = H(lambda)^{-1} v0 at
        '''
        Hv, x0, p0 = self.clf.hessian(), self.clf.critical(), self.cbf.critical()
        v0 = Hv.dot( p0 - x0 )

        H = self.compute_pencil( lambda_var )
        H_inv = np.linalg.inv(H)
        return H_inv.dot(v0)

    def compute_pencil(self, lambda_var):
        '''
        This function returns the value of the matrix pencil H(lambda) = lambda Hh - Hv
        '''
        Hv, Hh = self.clf.hessian(), self.cbf.hessian()
        return lambda_var*Hh - Hv

    # This function 
    def compute_compatibility(self):
        '''
        This function computes the polynomials of the rational compatibility funcion f(lambda). It assumes an invertible Hv.
        '''
        n = self.state_dim
        # Similarity transformation
        Hv, Hh = self.clf.hessian(), self.cbf.hessian()
        x0, p0 = self.clf.critical(), self.cbf.critical()
        v0 = Hv.dot( p0 - x0 )

        # Get the generalized Schur decomposition of the matrix pencil and compute the generalized eigenvalues
        schurHv, schurHh, alpha, beta, Q, Z = scipy.linalg.ordqz(Hv, Hh)
        for k in range(n):
            pencil_char_roots = alpha/beta

        # Assumption: Hv is invertible
        detHv = np.linalg.det(Hv)
        try:
            Hv_inv = np.linalg.inv(Hv)
            Hv_adj = detHv*Hv_inv
        except np.linalg.LinAlgError as error:
            print(error)
            return

        # Computes the pencil characteristic polynomial and denominator of f(\lambda)
        pencil_det = np.real(np.prod(pencil_char_roots))
        self.pencil_char = ( detHv/pencil_det ) * np.real(np.polynomial.polynomial.polyfromroots(pencil_char_roots))
        self.den_poly = np.polynomial.polynomial.polymul(self.pencil_char, self.pencil_char)

        # This computes the pencil adjugate expansion and the set of numerator vectors by the adapted Faddeev-LeVerrier algorithm.
        D = np.zeros([n, n, n])
        D[:][:][0] = pow(-1,n-1) * Hv_adj

        Omega = np.zeros([n,n])
        Omega[0,:] = D[:][:][0].dot(v0)

        for k in range(1,n):
            D[:][:][k] = np.matmul( Hv_inv, np.matmul(Hh, D[:][:][k-1]) - self.pencil_char[k]*np.eye(n) )
            Omega[k,:] = D[:][:][k].dot(v0)

        # Computes the numerator polynomial
        W = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                W[i,j] = np.inner(Hh.dot(Omega[i,:]), Omega[j,:])

        self.num_poly = np.polynomial.polynomial.polyzero
        for k in range(n):
            poly_term = np.polynomial.polynomial.polymul( W[:,k], np.eye(n)[:,k] )
            self.num_poly = np.polynomial.polynomial.polyadd(self.num_poly, poly_term)

        # Computes polynomial roots
        self.num_roots = np.polynomial.polynomial.polyroots(self.num_poly)
        self.den_roots = np.polynomial.polynomial.polyroots(self.den_poly)

        # Computes f(0)
        f0 = self.num_poly[0]/(self.pencil_char[0]**2)

        # Computes critical points
        self.dnum_poly = np.polynomial.polynomial.polyder(self.num_poly)
        self.dpencil_char = np.polynomial.polynomial.polyder(self.pencil_char)

        poly1 = np.polynomial.polynomial.polymul(self.dnum_poly, self.pencil_char)
        poly2 = 2*np.polynomial.polynomial.polymul(self.num_poly, self.dpencil_char)
        num_df = np.polynomial.polynomial.polysub( poly1, poly2 )

        self.critical_points = np.polynomial.polynomial.polyroots(num_df)
        self.critical_points = np.real(np.extract( self.critical_points.imag == 0.0, self.critical_points ))
        self.critical_values = self.f_values(self.critical_points)
        self.number_critical = len(self.critical_values)

    # Returns the values of f at args
    def f_values(self, args):
        numpoints = len(args)
        fvalues = np.zeros(numpoints)
        for k in range(numpoints):
            num_value = np.polynomial.polynomial.polyval( args[k], self.num_poly )
            pencil_char_value = np.polynomial.polynomial.polyval( args[k], self.pencil_char )
            fvalues[k] = num_value/(pencil_char_value**2)
        return fvalues

    # Returns the values of df at args
    def df_values(self, args):
        numpoints = len(args)
        dfvalues = np.zeros(numpoints)
        for k in range(numpoints):
            num_value = np.polynomial.polynomial.polyval( args[k], self.num_poly )
            dnum_value = np.polynomial.polynomial.polyval( args[k], self.dnum_poly )
            pencil_char_value = np.polynomial.polynomial.polyval( args[k], self.pencil_char )
            dpencil_char_value = np.polynomial.polynomial.polyval( args[k], self.dpencil_char )
            dfvalues[k] = dnum_value/(pencil_char_value**2) - 2*(num_value/(pencil_char_value**3))*dpencil_char_value
        return dfvalues


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
        if b.ndim != 1:
            raise Exception('b must be a 1-dim array.')
        if A.ndim == 2 and A.shape[0] != len(b):
            raise Exception('A and b must have the same number of lines.')
        self.A = A
        self.b = b
        self.dimension = A.shape[0]
        if A.ndim == 2:
            self.num_constraints = A.shape[1]
        else:
            self.num_constraints = 1

    # Returns the solution of the configured QP.
    def get_solution(self):
        self.solve_QP()
        return self.last_solution

    # Method for solving the configured QP using quadprog.
    def solve_QP(self):
        try:
            self.last_solution = solve_qp(P=self.P, q=self.q, G=self.A, h=self.b, solver="quadprog")
        except Exception as error:
            print(error)