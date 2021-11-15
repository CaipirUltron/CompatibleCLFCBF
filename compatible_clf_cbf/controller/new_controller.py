import numpy as np
import math, scipy

from compatible_clf_cbf.quadratic_program import QuadraticProgram
from compatible_clf_cbf.dynamic_systems import Quadratic, Integrator


class NewQPController():
    '''
    Class for the compatible QP controller.
    '''
    def __init__(self, plant, clf, ref_clf, cbf, gamma = [1.0, 1.0], alpha = [1.0, 1.0], p = [10.0, 10.0], dt = 0.001):

        # Dimensions and system model initialization
        self.plant = plant
        self.clf, self.ref_clf, self.cbf = clf, ref_clf, cbf

        self.state_dim = self.plant.n
        self.control_dim = self.plant.m
        self.sym_dim = int(( self.state_dim * ( self.state_dim + 1 ) )/2)
        self.skewsym_dim = int(( self.state_dim * ( self.state_dim - 1 ) )/2)
        self.sym_basis = Quadratic.symmetric_basis(self.state_dim)
        self.skewsym_basis = Quadratic.skewsymmetric_basis(self.state_dim)

        # Initialize rate CLF
        self.Vpi = 0.0
        self.gradient_Vpi = np.zeros(self.sym_dim)
        self.gradient_Vrot = np.zeros(self.skewsym_dim)

        # Initialize compatibility function parameters
        self.eigen_threshold = 0.000001
        self.schurHv = np.zeros([self.state_dim,self.state_dim])
        self.schurHh = np.zeros([self.state_dim,self.state_dim])
        self.Q = np.zeros([self.state_dim,self.state_dim])
        self.Z = np.zeros([self.state_dim,self.state_dim])
        self.pencil_dict = {}
        self.f_params_dict = {
            "epsilon": 1.2,
            "min_CLF_eigenvalue": 0.1,
        }
        self.compute_compatibility()

        # Parameters for the inner and outer QPs
        self.gamma, self.alpha, self.p = gamma, alpha, p

        # Parameters for the inner QP controller (QP1)
        self.QP1_dim = self.control_dim + 1
        P1 = np.eye(self.QP1_dim)
        P1[-1,-1] = self.p[0]
        q1 = np.zeros(self.QP1_dim)
        self.QP1 = QuadraticProgram(P=P1,q=q1)

        # Parameters for the outer QP controller (QP2)
        self.QP2_dim = self.sym_dim + 1
        P2 = np.eye(self.QP2_dim)
        P2[-1,-1] = self.p[1]
        q2 = np.zeros(self.QP2_dim)
        self.QP2 = QuadraticProgram(P=P2,q=q2)

        # Variable initialization
        self.ctrl_dt = dt
        self.V = 0.0
        self.h = 0.0
        self.u = np.zeros(self.control_dim)
        self.u_v = np.zeros(self.sym_dim)
        self.LgV = np.zeros(self.control_dim)
        self.Lgh = np.zeros(self.control_dim)

        # Initialize dynamic subsystems
        f_integrator, g_integrator = list(), list()
        state_string, ctrl_string = str(), str()
        EYE = np.eye(self.sym_dim)
        for k in range(self.sym_dim):
            f_integrator.append('0')
            g_integrator.append(EYE[k,:])
            state_string = state_string + 'pi' + str(k+1) + ', '
            ctrl_string = ctrl_string + 'dpi' + str(k+1) + ', '

        # Integrator sybsystem for the CLF parameters
        piv_init = Quadratic.sym2vector(self.clf.get_hessian())
        self.clf_dynamics = Integrator(piv_init,np.zeros(len(piv_init)))

    def get_control(self):
        '''
        Computes the QP solution.
        '''
        # Configure the QP
        a_clf, b_clf = self.get_clf_constraint()
        a_cbf, b_cbf = self.get_cbf_constraint()
        a_boundary_stable, b_boundary_stable = self.get_boundary_stability_constraint()

        a_rate, b_rate = self.get_rate_constraint()
        a_clf_rot, b_clf_rot = self.get_fix_rot_constraints()
        a_barriers, b_barriers = self.get_compatibility_constraints()

        A_nominal = np.vstack([ a_clf, a_cbf ])
        b_nominal = np.array([ b_clf, b_cbf ], dtype=float)

        # Adds compatibility constraints in case the trajectory is above the obstacle
        if (self.get_region() >= 0):
            A = np.vstack([ A_nominal, a_barriers ])
            b = np.hstack([ b_nominal, b_barriers ])
            # Adds boundary stability constraint in case the CBF is indefinite
            if (self.get_mode() <= 0):
                A = np.vstack([ A_nominal, a_boundary_stable ])
                b = np.hstack([ b_nominal, b_boundary_stable ])
        else:
            A = np.vstack([ A_nominal, a_rate ])
            b = np.hstack([ b_nominal, b_rate ])

        # Initializes constraints ('cleans' the QP)
        self.QP.initialize()

        # Sets the inequality constraints
        self.QP.set_constraints(A, b)

        # Sets the equality constraints (in case compatibility is needed)
        if (self.get_region() >= 0):
            self.QP.set_eq_constraints(a_clf_rot, b_clf_rot)

        # Solve QP
        QP_sol = self.QP.get_solution()

        self.u = QP_sol[0:self.control_dim]
        self.u_v = QP_sol[self.control_dim:self.control_dim+self.sym_dim]

        return self.u, self.u_v

    def get_clf_constraint(self):
        '''
        Sets the Lyapunov constraint.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()
        state = self.plant.get_state()

        # Lyapunov function and gradient
        self.V = self.clf.evaluate(state)
        self.nablaV = self.clf.get_gradient()

        # Lie derivatives
        self.LfV = self.nablaV.dot(f)
        self.LgV = g.T.dot(self.nablaV)

        # Gradient w.r.t. pi
        self.nablaV_pi = np.zeros(self.sym_dim)
        delta_x = ( state - self.clf.get_critical() ).reshape(self.state_dim,1)
        for k in range(self.sym_dim):
            self.nablaV_pi[k] = 0.5 * ( delta_x.T @ self.sym_basis[k] @ delta_x )[0,0]

        # Introduces instability in case the CBF is indefinite
        if (self.get_mode() <= 0) and (self.get_region() >= 0):
            eta = -1
        else:
            eta = +1

        # CLF constraint for the first QP
        a_clf = np.hstack([ eta*self.LgV, 0*eta*self.nablaV_pi, -1.0, 0.0 ])
        b_clf = - self.gamma[0]*self.V - eta*self.LfV

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
        self.h = self.cbf.evaluate(state)
        self.nablah = self.cbf.get_gradient()

        self.Lfh = self.nablah.dot(f)
        self.Lgh = g.T.dot(self.nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack([ self.Lgh, np.zeros(self.sym_dim), 0.0, 0.0 ])
        b_cbf = self.alpha[0] * self.h + self.Lfh

        return a_cbf, b_cbf

    def get_boundary_stability_constraint(self):
        '''
        Sets the constraint for boundary stability in case the CBF is indefinite.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()

        # Barrier function gradient
        self.nablah = self.cbf.get_gradient()

        self.Lfh = self.nablah.dot(f)
        self.Lgh = g.T.dot(self.nablah)

        # Boundary stability constraint for the first QP
        a_boundary_stable = np.hstack([ self.Lgh, np.zeros(self.sym_dim), -1.0, 0.0 ])
        b_boundary_stable = -self.Lfh

        return a_boundary_stable, b_boundary_stable

    def update_clf_dynamics(self, piv_ctrl):
        '''
        Integrates the dynamic system for the CLF Hessian matrix.
        '''
        self.clf_dynamics.set_control(piv_ctrl)
        self.clf_dynamics.actuate(self.ctrl_dt)
        pi_v = self.clf_dynamics.get_state()
        Hv = Quadratic.vector2sym(pi_v)

        self.clf.set_param(hessian = Hv)
        self.compute_compatibility()

    def get_rate_constraint(self):
        '''
        Sets the Lyapunov rate constraint.
        '''
        # Computes rate Lyapunov and gradient
        deltaHv = self.clf.get_hessian() - self.ref_clf.get_hessian()
        self.Vpi = 0.5 * np.trace( deltaHv @ deltaHv )
        for k in range(self.sym_dim):
            self.gradient_Vpi[k] = np.trace( deltaHv @ self.sym_basis[k] )

        # print("Vpi = " + str(self.Vpi))

        # Sets rate constraint
        a_clf_pi = np.hstack( [ np.zeros(self.control_dim), self.gradient_Vpi, 0.0, -1.0 ])
        b_clf_pi = -self.gamma[1] * self.Vpi

        return a_clf_pi, b_clf_pi

    def get_fix_rot_constraints(self):
        '''
        Sets the constraint for fixing the pencil eigenvectors.
        '''
        l = 0
        Jacobian_pi_omega = np.zeros([self.skewsym_dim, self.sym_dim])
        for i in range(self.state_dim):
            for j in range(i+1,self.state_dim):
                for k in range(self.sym_dim):
                    left_matrix = self.Q.T @ self.sym_basis[k] @ self.Z
                    Jacobian_pi_omega[l,k] = left_matrix[i,j]
                l = l + 1

        a_clf_rot = np.hstack( [ np.zeros([self.skewsym_dim, self.control_dim]), Jacobian_pi_omega, np.zeros([self.skewsym_dim, 2]) ])
        b_clf_rot = np.zeros(self.skewsym_dim)

        return a_clf_rot, b_clf_rot

    def get_compatibility_constraints(self):
        '''
        Sets the barrier constraints for compatibility.
        '''
        self.h_gamma1, gradient_h_gamma1 = self.first_compatibility_barrier()
        self.h_gamma2, gradient_h_gamma2 = self.second_compatibility_barrier()

        # Sets compatibility constraints
        a_cbf_gamma1 = -np.hstack([ np.zeros([self.num_positive_critical, self.control_dim]), gradient_h_gamma1, np.zeros([self.num_positive_critical, 2]) ])
        b_cbf_gamma1 = self.alpha[1]*self.h_gamma1

        a_cbf_gamma2 = -np.hstack([ np.zeros([3, self.control_dim]), gradient_h_gamma2, np.zeros([3,2]) ])
        b_cbf_gamma2 = self.alpha[1]*self.h_gamma2

        a_cbf_pi = np.vstack([ a_cbf_gamma1, a_cbf_gamma2 ])
        b_cbf_pi = np.hstack([ b_cbf_gamma1, b_cbf_gamma2 ])

        return a_cbf_pi, b_cbf_pi

    def first_compatibility_barrier(self):
        '''
        Computes first compatibility barrier constraint, for keeping the critical values of f above/below 1.
        '''
        # Get mode (convex or concave)
        mode = self.get_mode()

        # First barrier
        self.h_gamma1 = np.zeros(self.num_positive_critical)
        gradient_h_gamma1 = np.zeros([self.num_positive_critical, self.sym_dim])
        for k in range(self.num_positive_critical):

            # Barrier function
            self.h_gamma1[k] = mode * ( self.positive_critical_values[k] - np.exp( np.tanh(mode) * self.f_params_dict["epsilon"] ) )

            # Barrier function gradient
            v = self.v_values( self.positive_f_critical[k] )
            H = self.pencil_value( self.positive_f_critical[k] )
            vec_nabla_f = 2 * mode * ( np.linalg.inv(H) @ self.cbf.get_hessian() ) @ v
            for i in range(self.sym_dim):
                vec_i = self.sym_basis[i] @ ( v + self.cbf.get_critical() - self.clf.get_critical() )
                gradient_h_gamma1[k,i] = vec_nabla_f.T @ vec_i

        return self.h_gamma1, gradient_h_gamma1

    def second_compatibility_barrier(self):
        '''
        Computes second compatibility barrier constraint, for positive-type eigenvalues left and negative-type eigenvalues right on the f-function.
        Currently, it is in the form of a Matrix Barrier Function.
        '''
        # Matrix barrier function (MBF)
        Hv = self.clf.get_hessian()
        self.MCBF = Hv - self.f_params_dict["min_CLF_eigenvalue"] * np.eye(self.state_dim)
        eig_mbf, Q_mbf = np.linalg.eig(Hv)

        # print("Eigenvalues of Hv = " + str(eig_mbf))

        # Barrier functions for enforcing Sylvester's criterion
        self.first_minor = self.MCBF[0,0]
        self.second_minor = self.MCBF[1,1]
        self.third_minor = np.linalg.det(self.MCBF)

        # Barrier function gradients
        grad_first_minor = np.zeros(self.sym_dim)
        grad_second_minor = np.zeros(self.sym_dim)
        grad_third_minor = np.zeros(self.sym_dim)
        for i in range(self.sym_dim):
            grad_first_minor[i] = self.sym_basis[i][0,0]
            grad_second_minor[i] = self.sym_basis[i][1,1]
            grad_third_minor[i] = (Q_mbf[:,0].T @ self.sym_basis[i] @ Q_mbf[:,0]) * eig_mbf[1] + (Q_mbf[:,1].T @ self.sym_basis[i] @ Q_mbf[:,1]) * eig_mbf[0]

        self.h_gamma2 = np.array([ self.first_minor, self.second_minor, self.third_minor ]) - np.array([0.0, 0.0, 0.0])
        gradient_h_gamma2 = np.vstack([ grad_first_minor, grad_second_minor, grad_third_minor ])

        return self.h_gamma2, gradient_h_gamma2

    def get_mode(self):
        '''
        Computes the operation mode for a general n-th order system (for now, only for second order systems - determinant suffices).
        Positive for definite barrier Hessian, negative otherwise.
        '''
        Hh = self.cbf.get_hessian()
        return np.linalg.det(Hh)

    def get_region(self):
        '''
        Computes the region function: positive when is necessary to compatibilize, negative otherwise.
        Currently, we use a geometric method based on the relative position of the CLF-CBF.
        '''                
        state = self.plant.get_state()

        eig_cbf, Q_cbf = np.linalg.eig(self.cbf.get_hessian())
        x0 = self.clf.get_critical()
        p0 = self.cbf.get_critical()

        # Computes the normal vector defining the separatrix plane
        max_index = np.argmax(eig_cbf)
        if (Q_cbf[:,max_index] @ (x0 - p0)) >= 0:
            normal_vector = -Q_cbf[:,max_index]
        else:
            normal_vector = Q_cbf[:,max_index]

        return normal_vector @ ( state - p0 )

    def compute_eigenvalues(self):
        '''
        Computes pencil eigenvalues from Schur decomposition.
        '''
        alpha = np.diag(self.schurHv)
        beta = np.diag(self.schurHh)
        eigenvalues = alpha/beta
        sorted_args = np.argsort(eigenvalues)

        return eigenvalues, alpha, beta, sorted_args

    def compute_pencil(self, Hv, Hh):
        '''
        Given Hv and Hh, this method computes the generalized pencil eigenvalues and the pencil characteristic polynomial.
        '''
        self.schurHv, self.schurHh, alpha, beta, self.Q, self.Z = scipy.linalg.ordqz(Hv, Hh)

        pencil_eig, alpha, beta, sorted_args = self.compute_eigenvalues()

        # print("Q = " + str(self.Q))
        # print("Z = " + str(self.Z))
        # print("Pencil eig = " + str(pencil_eig))

        # Assumption: Hv is invertible => detHv != 0
        detHv = np.linalg.det(Hv)

        # Computes the pencil characteristic polynomial and denominator of f(\lambda)
        pencil_det = np.real(np.prod(pencil_eig))
        pencil_char = ( detHv/pencil_det ) * np.real(np.polynomial.polynomial.polyfromroots(pencil_eig))

        self.pencil_dict["eigenvalues"] = pencil_eig[sorted_args]
        self.pencil_dict["alpha"] = alpha[sorted_args]
        self.pencil_dict["beta"] = beta[sorted_args]
        self.pencil_dict["polar_eigenvalues"] = np.arctan(pencil_eig[sorted_args])
        self.pencil_dict["characteristic_polynomial"] = pencil_char

    def compute_f(self):
        '''
        This method computes rational function f.
        '''
        n = self.state_dim

        # Similarity transformation
        Hv, Hh = self.clf.get_hessian(), self.cbf.get_hessian()
        x0, p0 = self.clf.get_critical(), self.cbf.get_critical()
        v0 = Hv @ ( p0 - x0 )

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
        p0 = self.cbf.get_critical()
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
        Computes critical points of f.
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

        # Get positive critical points
        index, = np.where(self.f_critical > 0)
        self.positive_f_critical = self.f_critical[index]
        self.positive_critical_values = self.critical_values[index]
        self.num_positive_critical = len(self.positive_f_critical)

    def compute_compatibility(self):
        '''
        This function computes the polynomials of the rational compatibility funcion f(lambda). It assumes an invertible Hv.
        '''
        self.compute_f()
        self.compute_equilibrium()
        self.compute_f_critical()

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
        Hv, x0, p0 = self.clf.get_hessian(), self.clf.get_critical(), self.cbf.get_critical()
        v0 = Hv.dot( p0 - x0 )

        H = self.pencil_value( lambda_var )
        H_inv = np.linalg.inv(H)
        return H_inv.dot(v0)

    def pencil_value(self, lambda_var):
        '''
        This function returns the value of the matrix pencil H(lambda) = lambda Hh - Hv
        '''
        Hv, Hh = self.clf.get_hessian(), self.cbf.get_hessian()
        return lambda_var*Hh - Hv