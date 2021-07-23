from compatible_clf_cbf.dynamic_systems.dynamic_systems import QuadraticFunction
import numpy as np
import math, scipy
from qpsolvers import solve_qp
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
        self.eigen_threshold = 0.00001
        self.pencil_dict = {}
        self.f_dict = {}
        self.f_params_dict = {
            "minimum_gap": 1.0,
            "minimum_eigenvalue": 1.0,
        }
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
        self.h = 0.0

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
        # Computes the distance from the positive invariant set
        distance = self.distance_to_invariance(state)
        print("Distance = " + str(distance))

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
        if self.h_gamma >= 0 and self.h_positive >= 0: 
            convergence_rate = self.gamma[0]
        else:
            convergence_rate = self.gamma[0] * SimulateDynamics.sat( distance, 1.0 )
        b_clf = -convergence_rate * V - LfV

        return a_clf, b_clf

    def compute_cbf_constraint(self, state):
        '''
        Sets the barrier constraint for the inner loop controller.
        '''
        # Affine plant dynamics
        f = self._plant.compute_f(state)
        g = self._plant.compute_g(state)

        # Barrier function and gradient
        self.h = self.cbf(state)
        nablah = self.cbf.gradient(state)

        Lfh = nablah.dot(f)
        Lgh = g.T.dot(nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack( [ Lgh, 0.0 ])
        b_cbf = self.alpha[0] * self.h + Lfh

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
        b_outer = np.hstack([b_clf_pi, b_cbf_pi])

        self.outerQP.set_constraints(A = A_outer,b = b_outer)
        outerQP_sol = self.outerQP.get_solution()

        piv_control = outerQP_sol[0:self.symmetric_dim,]

        ##################################### Uncomment to turn off outer loop controller #####################################
        # piv_control = np.zeros(self.symmetric_dim)

        self.update_clf_dynamics(piv_control)

    def compute_rate_constraint(self, state):
        '''
        Sets the Lyapunov constraint for the outer loop controller.
        '''
        # Computes the inner product between gradients
        inner = np.inner( self.clf.gradient(state), self.cbf.gradient(state) )

        # Computes the distance from the positive invariant set
        distance = self.distance_to_invariance(state)

        # Computes rate Lyapunov and gradient
        deltaHv = self.clf.hessian() - self.ref_clf.hessian()
        self.Vpi = 0.5 * np.trace( np.matmul(deltaHv, deltaHv) )
        for k in range(self.symmetric_dim):
            self.gradient_Vpi[k] = np.trace( np.matmul( deltaHv, self.sym_basis[k]) )

        # Sets rate constraint
        a_clf_pi = np.hstack( [ self.gradient_Vpi, -0.0 ])
        approaching_obstacle = inner >= 0 and distance < 0.1
        # approaching_obstacle = inner >= 0
        if approaching_obstacle:
            b_clf_pi = math.inf
        else:
            b_clf_pi = -self.gamma[1] * self.Vpi

        return a_clf_pi, b_clf_pi

    def compute_compatibility_constraints(self, state):
        '''
        Sets the barrier constraints for the outer loop controller, ensuring compatibility.
        '''
        nablaV, nablah = self.clf.gradient(state), self.cbf.gradient(state)
        inner = np.inner( nablaV, nablah )
        distance = self.distance_to_invariance(state)

        # Constraint for keeping the eigenvalues positive
        pencil_eig = self.pencil_dict["eigenvalues"]
        Q = self.pencil_dict["left_eigenvectors"]
        Z = self.pencil_dict["right_eigenvectors"]
        
        # print("Eigen = " + str(pencil_eig))

        Hh = self.cbf.hessian()
        beta_0 = np.dot( Q[:,0], Hh.dot(Z[:,0]) )

        self.h_positive = pencil_eig[0] - self.f_params_dict["minimum_eigenvalue"]
        gradient_h_positive = np.zeros(self.symmetric_dim)
        for i in range(self.symmetric_dim):
            gradient_h_positive[i] = np.dot( Q[:,0], self.sym_basis[i].dot(Z[:,0]) ) / beta_0

        # h_gamma constraints
        self.h_gamma = np.zeros(self.number_critical)
        gradient_h_gamma = np.zeros([self.number_critical, self.symmetric_dim])
        for k in range(self.number_critical):
            self.h_gamma[k] = np.log( self.critical_values ) - self.f_params_dict["minimum_gap"]

            v = self.v_values( self.f_critical[k] )
            H = self.pencil_value( self.f_critical[k] )
            H_inv = np.linalg.inv(H)
            vec_nabla_f = 2 * (1/self.critical_values[k]) * np.matmul( self.cbf.hessian(), H_inv ).dot(v)
            for i in range(self.symmetric_dim):
                vec_i = self.sym_basis[i].dot( v + self.cbf.critical() - self.clf.critical() )
                gradient_h_gamma[k,i] = vec_nabla_f.dot(vec_i)

        # Sets compatibility constraints
        a_cbf_gamma = -np.hstack([ gradient_h_gamma, np.zeros([self.number_critical,1]) ])
        a_cbf_positive = -np.hstack([ gradient_h_positive, 0.0 ])
        a_cbf_pi = np.vstack([ a_cbf_gamma, a_cbf_positive ])

        approaching_obstacle = inner >= 0 and distance < 0.1
        # approaching_obstacle = inner >= 0
        if approaching_obstacle:
            b_cbf_pi = np.hstack([ self.alpha[1]*self.h_gamma, self.alpha[1]*self.h_positive ])
        else:
            b_cbf_pi = np.array([ math.inf, math.inf ])

        return a_cbf_pi, b_cbf_pi

    def compute_pencil(self, Hv, Hh):
        '''
        Given Hv and Hh, this method computes the generalized pencil eigenvalues and the pencil characteristic polynomial
        '''
        # Get the generalized Schur decomposition of the matrix pencil and compute the generalized eigenvalues
        schurHv, schurHh, alpha, beta, Q, Z = scipy.linalg.ordqz(Hv, Hh)
        pencil_eig = alpha/beta
        pencil_eig = np.real(np.extract( pencil_eig.imag == 0.0, pencil_eig ))
        sorted_args = np.argsort(pencil_eig)

        # Assumption: Hv is invertible => detHv != 0
        detHv = np.linalg.det(Hv)

        # Computes the pencil characteristic polynomial and denominator of f(\lambda)
        pencil_det = np.real(np.prod(pencil_eig))
        pencil_char = ( detHv/pencil_det ) * np.real(np.polynomial.polynomial.polyfromroots(pencil_eig))

        self.pencil_dict["eigenvalues"] = pencil_eig[sorted_args]
        self.pencil_dict["left_eigenvectors"] = Q[:,sorted_args]
        self.pencil_dict["right_eigenvectors"] = Z[:,sorted_args]
        self.pencil_dict["characteristic_polynomial"] = pencil_char

    def compute_cost_polynomial(self, v_bar):
        '''
        Computes the polynomial cost functional for the P+ distance, returning its real critical points.
        '''
        n = self.state_dim
        pencil_char = self.pencil_dict["characteristic_polynomial"]

        W = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                W[i,j] = np.inner(self.Omega[i,:], self.Omega[j,:])

        # Terms for cost polynomial
        term1 = np.polynomial.polynomial.polyzero
        for k in range(n):
            poly_term = np.polynomial.polynomial.polymul( W[:,k], np.eye(n)[:,k] )
            term1 = np.polynomial.polynomial.polyadd(term1, poly_term)
        yu = np.zeros(n)
        for k in range(n):
            yu[k] = np.inner( v_bar, self.Omega[k,:] )
        term2 = -2*np.polynomial.polynomial.polymul( pencil_char, yu )
        term3 = np.polynomial.polynomial.polymul( pencil_char, pencil_char )*(np.linalg.norm(v_bar)**2)

        # Compute the cost polynomial for the computation of critical points
        self.cost_polynomial = np.polynomial.polynomial.polyzero
        self.cost_polynomial = np.polynomial.polynomial.polyadd( self.cost_polynomial, term1 )
        self.cost_polynomial = np.polynomial.polynomial.polyadd( self.cost_polynomial, term2 )
        self.cost_polynomial = np.polynomial.polynomial.polyadd( self.cost_polynomial, term3 )

        self.dcost_polynomial = np.polynomial.polynomial.polyder( self.cost_polynomial )
        return np.sort( np.real( np.polynomial.polynomial.polyroots(self.dcost_polynomial) ) )

    def compute_f(self):
        '''
        This method computes rational function f 
        '''
        n = self.state_dim

        # Similarity transformation
        Hv, Hh = self.clf.hessian(), self.cbf.hessian()
        x0, p0 = self.clf.critical(), self.cbf.critical()
        v0 = Hv.dot( p0 - x0 )

        # Compute the pencil
        self.compute_pencil(Hv, Hh)
        pencil_eig = self.pencil_dict["eigenvalues"]
        pencil_char = self.pencil_dict["characteristic_polynomial"]

        # Compute denominator of f
        den_poly = np.polynomial.polynomial.polymul(pencil_char, pencil_char)

        detHv = np.linalg.det(Hv)
        try:
            Hv_inv = np.linalg.inv(Hv)
            Hv_adj = detHv*Hv_inv
        except np.linalg.LinAlgError as error:
            print(error)
            return

        # This computes the pencil adjugate expansion and the set of numerator vectors by the adapted Faddeev-LeVerrier algorithm.
        D = np.zeros([n, n, n])
        D[:][:][0] = pow(-1,n-1) * Hv_adj

        self.Omega = np.zeros([n,n])
        self.Omega[0,:] = D[:][:][0].dot(v0)
        for k in range(1,n):
            D[:][:][k] = np.matmul( Hv_inv, np.matmul(Hh, D[:][:][k-1]) - pencil_char[k]*np.eye(n) )
            self.Omega[k,:] = D[:][:][k].dot(v0)

        # Computes the numerator polynomial
        W = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                W[i,j] = np.inner(Hh.dot(self.Omega[i,:]), self.Omega[j,:])

        num_poly = np.polynomial.polynomial.polyzero
        for k in range(n):
            poly_term = np.polynomial.polynomial.polymul( W[:,k], np.eye(n)[:,k] )
            num_poly = np.polynomial.polynomial.polyadd(num_poly, poly_term)

        # Computes polynomial roots
        fzeros = np.real( np.polynomial.polynomial.polyroots(num_poly) )

        # Filters repeated poles from pencil_eig and numerator_roots
        repeated_poles = []
        for i in range( len(pencil_eig) ):
            for j in range( len(fzeros) ):
                if np.absolute(fzeros[j] - pencil_eig[i]) < self.eigen_threshold:
                    if np.any(repeated_poles == pencil_eig[i]):
                            break
                    else:
                        repeated_poles.append( pencil_eig[i] )
        repeated_poles = np.array( repeated_poles )

        print("Poles = " + str(pencil_eig))
        print("Zeros = " + str(fzeros))
        print("Repeated poles = " + str(repeated_poles))

        self.f_dict = {
            "denominator": den_poly,
            "numerator": num_poly,
            "poles": pencil_eig,
            "zeros": fzeros,
            "repeated_poles": repeated_poles
        }

    def compute_equilibrium(self):
        '''
        Compute equilibrium solutions and equilibrium points.
        '''
        p0 = self.cbf.critical()
        solution_poly = np.polynomial.polynomial.polysub( self.f_dict["numerator"], self.f_dict["denominator"] )
        
        equilibrium_solutions = np.polynomial.polynomial.polyroots(solution_poly)
        equilibrium_solutions = np.real(np.extract( equilibrium_solutions.imag == 0.0, equilibrium_solutions ))
        equilibrium_solutions = np.concatenate((equilibrium_solutions, self.f_dict["repeated_poles"]))

        # Extract positive solutions and sort array
        self.equilibrium_solutions = np.sort( np.extract( equilibrium_solutions > 0, equilibrium_solutions ) )

        # Compute equilibrium points from equilibrium solutions
        self.equilibrium_points = np.zeros([self.state_dim,len(self.equilibrium_solutions)])
        for k in range(len(self.equilibrium_solutions)):
            if all(np.absolute(self.equilibrium_solutions[k] - self.pencil_dict["eigenvalues"]) > self.eigen_threshold ):
                self.equilibrium_points[:,k] = self.v_values( self.equilibrium_solutions[k] ) + p0

    def compute_f_critical(self):
        '''
        Computes critical points of f
        '''
        dnum_poly = np.polynomial.polynomial.polyder(self.f_dict["numerator"])
        dpencil_char = np.polynomial.polynomial.polyder(self.pencil_dict["characteristic_polynomial"])

        poly1 = np.polynomial.polynomial.polymul(dnum_poly, self.pencil_dict["characteristic_polynomial"])
        poly2 = 2*np.polynomial.polynomial.polymul(self.f_dict["numerator"], dpencil_char)
        num_df = np.polynomial.polynomial.polysub( poly1, poly2 )

        self.f_critical = np.polynomial.polynomial.polyroots(num_df)
        self.f_critical = np.real(np.extract( self.f_critical.imag == 0.0, self.f_critical ))
        self.critical_values = self.f_values(self.f_critical)
        self.number_critical = len(self.critical_values)

    def compute_compatibility(self):
        '''
        This function computes the polynomials of the rational compatibility funcion f(lambda). It assumes an invertible Hv.
        '''
        self.compute_f()
        self.compute_equilibrium()
        self.compute_f_critical()

    def distance_to_invariance(self, state):
        '''
        This function computes the minimum distance from the state trajectory to the danger subset of P+.
        The danger subset is the one at the open interval (lambda_1, lambda_n), that is, the interval that could contain stable equilibrium solutions.
        '''
        v_bar = state - self.cbf.critical()
        pencil_eig = self.pencil_dict["eigenvalues"]
        candidates = self.compute_cost_polynomial(v_bar)
        
        # Filters the solution candidates. Distance must be computed from a possibly stable equilibrium point.
        valid_candidates = []
        for candidate in candidates:
            valid_interval = candidate > pencil_eig[0] and candidate < pencil_eig[1]
            non_singular = all( np.absolute(pencil_eig - candidate) > self.eigen_threshold )
            if valid_interval and non_singular:
                valid_candidates.append( candidate )

        # Computes the best cost among the valid solution candidates
        distance = math.inf
        for c in valid_candidates:
            c_cost = self.cost_value(c, v_bar)
            if c_cost < distance:
                distance = c_cost

        return distance

    def f_values(self, args):
        '''
        Returns the values of f.
        '''
        numpoints = len(args)
        fvalues = np.zeros(numpoints)
        for k in range(numpoints):
            num_value = np.polynomial.polynomial.polyval( args[k], self.f_dict["numerator"] )
            pencil_char_value = np.polynomial.polynomial.polyval( args[k], self.pencil_dict["characteristic_polynomial"] )
            fvalues[k] = num_value/(pencil_char_value**2)
        return fvalues

    def v_values( self, lambda_var ):
        '''
        This function returns the value of vector v(lambda) = H(lambda)^{-1} v0
        '''
        Hv, x0, p0 = self.clf.hessian(), self.clf.critical(), self.cbf.critical()
        v0 = Hv.dot( p0 - x0 )

        H = self.pencil_value( lambda_var )
        H_inv = np.linalg.inv(H)
        return H_inv.dot(v0)

    def cost_value(self, lambda_bar, v_bar):
        return np.linalg.norm( self.v_values(lambda_bar) - v_bar )

    def pencil_value(self, lambda_var):
        '''
        This function returns the value of the matrix pencil H(lambda) = lambda Hh - Hv
        '''
        Hv, Hh = self.clf.hessian(), self.cbf.hessian()
        return lambda_var*Hh - Hv


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

    def set_cost(self, P, q):
        '''
        Set cost of the type x'Px + q'x 
        '''
        if ( P.ndim != 2 or q.ndim != 1 ):
            raise Exception('P must be a 2-dim array and q must be a 1-dim array.')
        if P.shape[0] != P.shape[1]:
            raise Exception('P must be a square matrix.')
        if P.shape[0] != len(q):
            raise Exception('A and b must have the same number of lines.')
        self.P = P
        self.q = q
        self.dimension = len(q)

    def set_constraints(self, A, b):
        '''
        Set constraints of the type A x <= b
        '''
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

    def get_solution(self):
        '''
        Returns the solution of the configured QP.
        '''
        self.solve_QP()
        return self.last_solution

    def solve_QP(self):
        '''
        Method for solving the configured QP using quadprog.
        '''
        try:
            self.last_solution = solve_qp(P=self.P, q=self.q, G=self.A, h=self.b, solver="quadprog")
        except Exception as error:
            print(error)