import numpy as np
import math, scipy

from compatible_clf_cbf.quadratic_program import QuadraticProgram
from compatible_clf_cbf.dynamic_systems import AffineSystem
from compatible_clf_cbf.dynamic_simulation import SimulateDynamics
from compatible_clf_cbf.dynamic_systems.dynamic_systems import QuadraticFunction


class NewQPController():
    '''
    Class for the compatible QP controller.
    '''
    def __init__(self, plant, clf, ref_clf, cbf, gamma = [1.0, 1.0], alpha = [1.0, 1.0], p = [10.0, 10.0], dt = 0.001):

        # Dimensions and system model initialization
        self._plant = plant
        self.clf, self.ref_clf, self.cbf = clf, ref_clf, cbf

        self.state_dim = self._plant.state_dim
        self.control_dim = self._plant.control_dim
        self.sym_dim = int(( self.state_dim * ( self.state_dim + 1 ) )/2)
        self.sym_basis = QuadraticFunction.symmetric_basis(self.state_dim)

        # Initialize rate CLF
        self.Vpi = 0.0
        self.gradient_Vpi = np.zeros(self.sym_dim)

        # Initialize compatibility function
        self.eigen_threshold = 0.00001
        self.pencil_dict = {}
        self.f_dict = {}
        self.f_params_dict = {
            "minimum_gap": 0.5,
            "minimum_eigenvalue": 0.1,
        }
        self.compute_compatibility()

        # Parameters for the inner and outer QPs
        self.gamma, self.alpha = gamma, alpha

        # Parameters for QP controller based on the new theorem
        self.p = p
        self.QP_dim = self.control_dim + self.sym_dim + 1
        P = np.eye(self.QP_dim)
        for i in range(0,self.sym_dim):
            P[self.control_dim+i,self.control_dim+i] = self.p[0]
        P[-1,-1] = self.p[1]

        q = np.zeros(self.QP_dim)
        self.QP = QuadraticProgram(P=P,q=q)

        # Variable initialization
        self.ctrl_dt = dt
        self.V = 0.0
        self.h = 0.0
        self.u = np.zeros(self.control_dim)
        self.u_pi = np.zeros(self.sym_dim)
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
        piv_integrator = AffineSystem(state_string, ctrl_string, f_integrator, *g_integrator)
        piv_init = QuadraticFunction.sym2vector(self.clf.hessian())
        self.clf_dynamics = SimulateDynamics(piv_integrator, piv_init)

    def get_control(self, state):
        '''
        Computes the QP solution.
        '''
        a_clf, b_clf = self.get_clf_constraint(state)
        a_cbf, b_cbf = self.get_cbf_constraint(state)
        a_rate, b_rate = self.get_rate_constraint(state)
        A_barriers, b_barriers = self.get_compatibility_constraints(state)

        # Stacking the CLF and CBF constraints
        A1 = np.vstack([ a_clf, a_cbf, a_rate ])
        b1 = np.array([ b_clf, b_cbf, b_rate ], dtype=float)

        A = np.vstack([ A1, A_barriers ])
        b = np.hstack([ b1, b_barriers ])

        # Solve inner QP
        self.QP.set_constraints(A, b)
        QP_sol = self.QP.get_solution()

        self.u = QP_sol[0:self.control_dim]
        self.u_pi = QP_sol[self.control_dim:self.control_dim+self.sym_dim]

        return self.u, self.u_pi

    def get_clf_constraint(self, state):
        '''
        Sets the Lyapunov constraint for the inner loop controller.
        '''
        # Affine plant dynamics
        f = self._plant.compute_f(state)
        g = self._plant.compute_g(state)

        # Lyapunov function and gradient
        self.V = self.clf(state)
        self.nablaV = self.clf.gradient(state)

        # Lie derivatives
        self.LfV = self.nablaV.dot(f)
        self.LgV = g.T.dot(self.nablaV)

        # Gradient w.r.t. pi
        self.nablaV_pi = np.zeros(self.sym_dim)
        delta_x = ( state - self.clf.critical_point ).reshape(self.state_dim,1)
        for k in range(self.sym_dim):
            self.nablaV_pi[k] = 0.5 * ( delta_x.T @ self.sym_basis[k] @ delta_x )[0,0]

        # CLF contraint for the QP
        delta = np.heaviside(float(self.get_selection()),0.0)
        a_clf = np.hstack( [ self.LgV, self.nablaV_pi*delta, 0.0 ])
        # b_clf = -self.gamma[0] * self.V - self.LfV - self.nablaV_pi.dot(self.u_pi)*( 1.0 - delta )
        b_clf = -self.gamma[0] * self.V - self.LfV

        return a_clf, b_clf

    def get_cbf_constraint(self, state):
        '''
        Sets the barrier constraint.
        '''
        # Affine plant dynamics
        f = self._plant.compute_f(state)
        g = self._plant.compute_g(state)

        # Barrier function and gradient
        self.h = self.cbf(state)
        self.nablah = self.cbf.gradient(state)

        self.Lfh = self.nablah.dot(f)
        self.Lgh = g.T.dot(self.nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack( [ self.Lgh, np.zeros(self.sym_dim), 0.0 ])
        b_cbf = self.alpha[0] * self.h + self.Lfh

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

    def get_rate_constraint(self, state):
        '''
        Sets the Lyapunov constraint for the outer loop controller.
        '''
        # Computes rate Lyapunov and gradient
        deltaHv = self.clf.hessian() - self.ref_clf.hessian()
        self.Vpi = 0.5 * np.trace( deltaHv @ deltaHv )
        for k in range(self.sym_dim):
            self.gradient_Vpi[k] = np.trace( deltaHv @ self.sym_basis[k] )

        # Sets rate constraint
        a_clf_pi = np.hstack( [ np.zeros(self.control_dim), self.gradient_Vpi, -1.0 ])
        b_clf_pi = -self.gamma[1] * self.Vpi

        print("Vpi = " + str(self.Vpi))

        return a_clf_pi, b_clf_pi

    def get_compatibility_constraints(self, state):
        '''
        Sets the barrier constraints for the outer loop controller, ensuring compatibility.
        '''
        # Constraint for keeping the eigenvalues positive
        pencil_eig = self.pencil_dict["eigenvalues"]
        Q = self.pencil_dict["left_eigenvectors"]
        Z = self.pencil_dict["right_eigenvectors"]
        
        Hh = self.cbf.hessian()
        beta_0 = np.dot( Q[:,0], Hh.dot(Z[:,0]) )

        self.h_positive = pencil_eig[0] - self.f_params_dict["minimum_eigenvalue"]
        gradient_h_positive = np.zeros(self.sym_dim)
        for i in range(self.sym_dim):
            gradient_h_positive[i] = np.dot( Q[:,0], self.sym_basis[i].dot(Z[:,0]) ) / beta_0

        # h_gamma constraints
        self.h_gamma = np.zeros(self.number_critical)
        gradient_h_gamma = np.zeros([self.number_critical, self.sym_dim])
        for k in range(self.number_critical):
            self.h_gamma[k] = np.log( self.critical_values ) - self.f_params_dict["minimum_gap"]

            v = self.v_values( self.f_critical[k] )
            H = self.pencil_value( self.f_critical[k] )
            H_inv = np.linalg.inv(H)
            vec_nabla_f = 2 * (1/self.critical_values[k]) * np.matmul( self.cbf.hessian(), H_inv ).dot(v)
            for i in range(self.sym_dim):
                vec_i = self.sym_basis[i].dot( v + self.cbf.critical() - self.clf.critical() )
                gradient_h_gamma[k,i] = vec_nabla_f.dot(vec_i)

        # Applies selection function
        if self.get_selection() >= 0:
            term = 0.0
        else:
            term = -100

        print("Term = " + str(self.get_selection()))

        # Sets compatibility constraints
        a_cbf_gamma = -np.hstack([ np.zeros([self.number_critical, self.control_dim]), gradient_h_gamma, np.zeros([self.number_critical, 1]) ])
        b_cbf_gamma = self.alpha[1]*self.h_gamma - term

        a_cbf_positive = -np.hstack([ np.zeros(self.control_dim), gradient_h_positive, 0.0 ])
        b_cbf_positive = self.alpha[1]*self.h_positive - term

        a_cbf_pi = np.vstack([ a_cbf_gamma, a_cbf_positive ])
        b_cbf_pi = np.hstack([ b_cbf_gamma, b_cbf_positive ])

        return a_cbf_pi, b_cbf_pi

    def get_selection(self):
        '''
        Computes the selection function: positive in the boundary convergence area, negative otherwise.
        '''
        LgV2 = self.LgV.dot(self.LgV)
        Lgh2 = self.Lgh.dot(self.Lgh)
        LgVLgh = self.LgV.dot(self.Lgh)
                
        return self.V * LgVLgh - self.h * math.sqrt(LgV2)*math.sqrt(Lgh2)
        # return self.V * LgVLgh

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
        Hv, x0, p0 = self.clf.hessian(), self.clf.critical(), self.cbf.critical()
        v0 = Hv.dot( p0 - x0 )

        H = self.pencil_value( lambda_var )
        H_inv = np.linalg.inv(H)
        return H_inv.dot(v0)

    def pencil_value(self, lambda_var):
        '''
        This function returns the value of the matrix pencil H(lambda) = lambda Hh - Hv
        '''
        Hv, Hh = self.clf.hessian(), self.cbf.hessian()
        return lambda_var*Hh - Hv