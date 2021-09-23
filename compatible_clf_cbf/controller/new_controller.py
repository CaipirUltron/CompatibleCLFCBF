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
        self.sym_basis = Quadratic.symmetric_basis(self.state_dim)

        # Initialize rate CLF
        self.Vpi = 0.0
        self.gradient_Vpi = np.zeros(self.sym_dim)

        # Initialize compatibility function
        self.eigen_threshold = 0.00001
        self.pencil_dict = {}
        self.f_dict = {}
        self.f_params_dict = {
            "epsilon": 0.1,
            "Kappa": 100
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
        piv_init = Quadratic.sym2vector(self.clf.get_hessian())
        self.clf_dynamics = Integrator(piv_init,np.zeros(len(piv_init)))

    def get_control(self):
        '''
        Computes the QP solution.
        '''
        a_clf, b_clf = self.get_clf_constraint()
        a_cbf, b_cbf = self.get_cbf_constraint()
        a_rate, b_rate = self.get_rate_constraint()
        A_barriers, b_barriers = self.get_compatibility_constraints()

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

        # CLF contraint for the QP
        delta = np.heaviside(float(self.get_selection()),0.0)
        a_clf = np.hstack( [ self.LgV, self.nablaV_pi*delta, 0.0 ])
        # b_clf = -self.gamma[0] * self.V - self.LfV - self.nablaV_pi.dot(self.u_pi)*( 1.0 - delta )
        b_clf = - self.gamma[0] * np.abs(self.V) - self.LfV

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
        a_cbf = -np.hstack( [ self.Lgh, np.zeros(self.sym_dim), 0.0 ])
        b_cbf = self.alpha[0] * self.h + self.Lfh

        return a_cbf, b_cbf

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

        # Sets rate constraint
        a_clf_pi = np.hstack( [ np.zeros(self.control_dim), self.gradient_Vpi, -1.0 ])
        b_clf_pi = -self.gamma[1] * self.Vpi

        # print("Vpi = " + str(self.Vpi))

        return a_clf_pi, b_clf_pi

    def get_compatibility_constraints(self):
        '''
        Sets the barrier constraints for compatibility.
        '''
        self.h_gamma1, gradient_h_gamma1 = self.first_compatibility_barrier()
        self.h_gamma2, gradient_h_gamma2 = self.second_compatibility_barrier()

        # Applies selection function
        if self.get_selection() >= 0:
            kappa_term = 0.0
        else:
            kappa_term = self.f_params_dict["Kappa"]

        # Sets compatibility constraints
        a_cbf_gamma1 = -np.hstack([ np.zeros([self.number_critical, self.control_dim]), gradient_h_gamma1, np.zeros([self.number_critical, 1]) ])
        b_cbf_gamma1 = self.alpha[1]*self.h_gamma1 + kappa_term*np.ones(self.number_critical)

        a_cbf_gamma2 = -np.hstack([ np.zeros([self.state_dim, self.control_dim]), gradient_h_gamma2, np.zeros([self.state_dim, 1]) ])
        b_cbf_gamma2 = self.alpha[1]*self.h_gamma2

        a_cbf_pi = np.vstack([ a_cbf_gamma1, a_cbf_gamma2 ])
        b_cbf_pi = np.hstack([ b_cbf_gamma1, b_cbf_gamma2 ])

        return a_cbf_pi, b_cbf_pi

    def first_compatibility_barrier(self):
        '''
        Computes first compatibility barrier constraint, for keeping the critical values of f below 1.
        '''
        pencil_eig = self.pencil_dict["eigenvalues"]
        Q = self.pencil_dict["left_eigenvectors"]
        Z = self.pencil_dict["right_eigenvectors"]
        
        Hh = self.cbf.get_hessian()

        # First barrier
        self.h_gamma1 = np.zeros(self.number_critical)
        gradient_h_gamma1 = np.zeros([self.number_critical, self.sym_dim])
        for k in range(self.number_critical):

            index, = np.where(pencil_eig<=self.f_critical[k])
            closest_eig_index = len(pencil_eig[index])-1

            beta_k = np.dot( Q[:,closest_eig_index], Hh.dot(Z[:,closest_eig_index]) )
            epsilon = self.f_params_dict["epsilon"]
            self.h_gamma1[k] = beta_k * np.log( self.critical_values[k] * np.exp( -np.tanh(beta_k) * epsilon ) )

            v = self.v_values( self.f_critical[k] )
            H = self.pencil_value( self.f_critical[k] )
            H_inv = np.linalg.inv(H)
            vec_nabla_f = 2 * beta_k * (1/self.critical_values[k]) * np.matmul( Hh, H_inv ).dot(v)
            for i in range(self.sym_dim):
                vec_i = self.sym_basis[i].dot( v + self.cbf.get_critical() - self.clf.get_critical() )
                gradient_h_gamma1[k,i] = vec_nabla_f.dot(vec_i)

        return self.h_gamma1, gradient_h_gamma1

    def second_compatibility_barrier(self):
        '''
        Computes second compatibility barrier constraint, for positive-type eigenvalues left and negative-type eigenvalues right on the f-function.
        '''
        polar_pencil_eigs = self.pencil_dict["polar_eigenvalues"]
        Q = self.pencil_dict["left_eigenvectors"]
        Z = self.pencil_dict["right_eigenvectors"]
        
        Hh = self.cbf.get_hessian()
        mode = self.get_mode()

        # Second barrier
        self.h_gamma2 = np.zeros(self.state_dim)
        gradient_h_gamma2 = np.zeros([self.state_dim, self.sym_dim])
        for k in range(self.state_dim):
            beta_k = np.dot( Q[:,k], Hh.dot(Z[:,k]) )
            self.h_gamma2[k] = mode * beta_k * polar_pencil_eigs[k]

            for i in range(self.sym_dim):
                L_i = self.sym_basis[i]
                gradient_h_gamma2[k,i] = mode * np.cos(polar_pencil_eigs[k]) * ( Q[:,k].T @ L_i @ Z[:,k] ) 

        return self.h_gamma2, gradient_h_gamma2

    def get_mode(self):
        '''
        Computes the operation mode for a general n-th order system (for now, only for second order systems - determinant suffices).
        Positive for definite barrier Hessian, negative otherwise.
        '''
        Hh = self.cbf.get_hessian()
        return np.linalg.det(Hh)

    def get_selection(self):
        '''
        Computes the selection function: positive when is necessary to compatibilize, negative otherwise.
        '''
        LgV2 = self.LgV.dot(self.LgV)
        Lgh2 = self.Lgh.dot(self.Lgh)
        LgVLgh = self.LgV.dot(self.Lgh)
                
        return self.get_mode() * ( self.V * LgVLgh - self.h * math.sqrt(LgV2)*math.sqrt(Lgh2) )
        # return self.get_mode() * LgVLgh

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
        self.pencil_dict["polar_eigenvalues"] = np.arctan(pencil_eig[sorted_args])
        self.pencil_dict["left_eigenvectors"] = Q[:,sorted_args]
        self.pencil_dict["right_eigenvectors"] = Z[:,sorted_args]
        self.pencil_dict["characteristic_polynomial"] = pencil_char

    def compute_f(self):
        '''
        This method computes rational function f 
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