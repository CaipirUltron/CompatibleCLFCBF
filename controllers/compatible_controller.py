import scipy
from scipy import signal
import numpy as np
from quadratic_program import QuadraticProgram


class CompatibleQPController():
    '''
    Class for the compatible QP controller.
    '''
    def __init__(self, plant, clf, ref_clf, cbfs, gamma = [1.0, 1.0], alpha = [1.0, 1.0], p = [10.0, 10.0], dt = 0.001):

        # Dimensions and system model initialization
        self.plant = plant
        self.clf, self.ref_clf = clf, ref_clf
        self.cbfs = cbfs
        self.active_cbf = None
        self.num_cbfs = len(self.cbfs)

        self.state_dim = self.plant.n
        self.control_dim = self.plant.m
        self.sym_dim = int(( self.state_dim * ( self.state_dim + 1 ) )/2)
        self.skewsym_dim = int(( self.state_dim * ( self.state_dim - 1 ) )/2)

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

        self.mode_log = []                   # mode = 1 for compatibility, mode = 0 for rate
        self.pencil_dict = {}
        self.f_params_dict = {
            "epsilon": 0.4,
            "min_CLF_eigenvalue": 0.2
        }
        self.clf.set_epsilon(self.f_params_dict["min_CLF_eigenvalue"])
        self.ref_clf.set_epsilon(self.f_params_dict["min_CLF_eigenvalue"])

        # Parameters for the inner and outer QPs
        self.gamma, self.alpha, self.p = gamma, alpha, p

        # Parameters for the inner QP controller (QP1)
        self.QP1_dim = self.control_dim + 1
        P1 = np.eye(self.QP1_dim)
        P1[-1,-1] = self.p[0]
        q1 = np.zeros(self.QP1_dim)
        self.QP1 = QuadraticProgram(P=P1,q=q1)
        self.QP1_sol = np.zeros(self.QP1_dim)

        # Parameters for the outer QP controller (QP2)
        self.QP2_dim = self.sym_dim + 1
        P2 = np.eye(self.QP2_dim)
        P2[-1,-1] = self.p[1]
        q2 = np.zeros(self.QP2_dim)
        self.QP2 = QuadraticProgram(P=P2,q=q2)
        self.QP2_sol = np.zeros(self.QP2_dim)

        # Variable initialization
        self.ctrl_dt = dt
        self.V = 0.0
        self.u = np.zeros(self.control_dim)
        self.u_v = np.zeros(self.sym_dim)

        self.compute_compatibility()

    def get_control(self):
        '''
        Computes the solution of the inner QP.
        '''
        # Configure constraints
        A, b = self.get_clf_constraint()

        for cbf in self.cbfs:
            a_cbf, b_cbf = self.get_cbf_constraint(cbf)
            A = np.vstack( [ A, a_cbf ])
            b = np.hstack( [ b, b_cbf ])

        # Solve inner loop QP
        self.QP1.initialize()
        self.QP1.set_inequality_constraints(A, b)
        self.QP1_sol = self.QP1.get_solution()
        self.u = self.QP1_sol[0:self.control_dim]

        return self.u

    def get_clf_control(self):
        '''
        Computes the solution of the outer QP.
        '''
        self.QP2.initialize()
        a_rate, b_rate = self.get_rate_constraint()

        # Adds compatibility/rate constraints
        if self.active_cbf:
            '''
            Compatibility constraints are added if an active CBF exists
            '''
            self.mode_log.append(1.0)
            a_clf_rot, b_clf_rot = self.get_eigenvector_constraints()
            a_cbf_pi, b_cbf_pi = self.get_compatibility_constraints()
            A_outer = np.vstack([a_rate, a_cbf_pi])
            b_outer = np.hstack([b_rate, b_cbf_pi])
            self.QP2.set_equality_constraints(a_clf_rot, b_clf_rot)
        else:
            '''
            Instead, rate constraints are added if no active CBF exists
            '''
            self.mode_log.append(0.0)
            # a_rate, b_rate = self.get_rate_constraint()
            A_outer = a_rate
            b_outer = np.array([ b_rate ])

        self.QP2.set_inequality_constraints(A_outer, b_outer)

        # Solve outer loop QP
        QP2_sol = self.QP2.get_solution()
        self.u_v = QP2_sol[0:self.sym_dim]

        return self.u_v

    def get_clf_constraint(self):
        '''
        Sets the Lyapunov constraint.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()
        state = self.plant.get_state()

        # Lyapunov function and gradient
        self.V = self.clf.evaluate_function(*state)[0]
        nablaV = self.clf.evaluate_gradient(*state)[0]

        # Lie derivatives
        LfV = nablaV.dot(f)
        LgV = g.T.dot(nablaV)

        # Gradient w.r.t. pi
        partial_Hv = self.clf.get_partial_Hv()
        self.nablaV_pi = np.zeros(self.sym_dim)
        delta_x = ( state - self.clf.get_critical() ).reshape(self.state_dim,1)
        for k in range(self.sym_dim):
            self.nablaV_pi[k] = 0.5 * ( delta_x.T @ partial_Hv[k] @ delta_x )[0,0]

        # CLF constraint for the first QP
        a_clf = np.hstack([ LgV, -1.0 ])
        b_clf = - self.gamma[0]*self.V - LfV

        return a_clf, b_clf

    def get_cbf_constraint(self, cbf):
        '''
        Sets the barrier constraint.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()
        state = self.plant.get_state()

        # Barrier function and gradient
        h = cbf.evaluate_function(*state)[0]
        nablah = cbf.evaluate_gradient(*state)[0]

        Lfh = nablah.dot(f)
        Lgh = g.T.dot(nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack([ Lgh, 0.0 ])
        b_cbf = self.alpha[0] * h + Lfh

        return a_cbf, b_cbf

    def update_clf_dynamics(self, piv_ctrl):
        '''
        Integrates the dynamic system for the CLF Hessian matrix.
        '''
        self.clf.update(piv_ctrl, self.ctrl_dt)
        self.compute_compatibility()

    # def update_cbf_dynamics(self, pih_ctrl):
    #     '''
    #     Integrates the dynamic system for the CBF Hessian matrix.
    #     '''
    #     self.active_cbf.update(pih_ctrl, self.ctrl_dt)
    #     self.compute_compatibility()

    def compute_active_cbf(self):
        '''
        Computes the active CBF.
        '''
        index = self.active_cbf_index()
        if index >= 0:
            self.active_cbf = self.cbfs[index]
        else:
            self.active_cbf = None

    def active_cbf_index(self):
        '''
        Returns the index of the current active CBF, if only one CBF is active.
        Returns -1 otherwise.
        '''
        cbf_constraints = []
        for cbf in self.cbfs:
            a_cbf, b_cbf = self.get_cbf_constraint(cbf)
            cbf_constraints.append( -a_cbf @ self.QP1_sol + b_cbf )

        arr = np.array(cbf_constraints) <= np.array([ 0.000001 for _ in range(len(self.cbfs)) ])

        count_sum = False
        for i in range(len(arr)):
            count_mult = True
            for j in range(len(arr)):
                if i != j:
                    count_mult = count_mult and not(arr[j])
            count_sum = count_sum or count_mult

        if count_sum:
            for index in range(len(arr)):
                if arr[index] == True:
                    return index
        
        return -1

    def get_rate_constraint(self):
        '''
        Sets the Lyapunov rate constraint.
        '''
        # Computes rate Lyapunov and gradient
        deltaHv = self.clf.get_hessian() - self.ref_clf.get_hessian()
        self.Vpi = 0.5 * np.trace( deltaHv @ deltaHv )
        partial_Hv = self.clf.get_partial_Hv()
        for k in range(self.sym_dim):
            self.gradient_Vpi[k] = np.trace( deltaHv @ partial_Hv[k] )

        # Sets rate constraint
        a_clf_pi = np.hstack( [ self.gradient_Vpi, -1.0 ])
        b_clf_pi = -self.gamma[1] * self.Vpi

        return a_clf_pi, b_clf_pi

    def get_eigenvector_constraints(self):
        '''
        Sets the constraint for fixing the pencil eigenvectors.
        '''
        JacobianV = np.zeros([self.skewsym_dim, self.sym_dim])
        Z = self.pencil_dict["eigenvectors"]
        partial_Hv = self.clf.get_partial_Hv()

        for l in range(self.sym_dim):
            diag_matrix = Z.T @ partial_Hv[l] @ Z
            m = 0
            for i in range(self.state_dim):
                for j in range(self.state_dim):
                    if i < j:
                        JacobianV[m,l] = diag_matrix[i,j]
                        m += 1

        a_clf_rot = np.hstack( [ JacobianV, np.zeros([self.skewsym_dim, 1]) ])
        b_clf_rot = np.zeros(self.skewsym_dim)

        return a_clf_rot, b_clf_rot

    def get_compatibility_constraints(self):
        '''
        Sets the barrier constraints for compatibility.
        '''
        self.h_gamma, gradient_h_gamma = self.compatibility_barrier()
        a_cbf_pi = -np.hstack([ gradient_h_gamma, np.zeros([self.state_dim-1, 1]) ])
        b_cbf_pi = self.alpha[1]*self.h_gamma

        return a_cbf_pi, b_cbf_pi

    def compatibility_barrier(self):
        '''
        Computes compatibility barrier constraint, for keeping the critical values of f above 1.
        '''
        Hv = self.clf.get_hessian()
        partial_Hv = self.clf.get_partial_Hv()
        x0 = self.clf.get_critical()

        p0 = self.active_cbf.get_critical()

        Z = self.pencil_dict["eigenvectors"]
        pencil_eig = self.pencil_dict["eigenvalues"]
        v0 = Hv @ ( p0 - x0 )
        self.cbfs
        # Compatibility barrier
        h_gamma = np.zeros(self.state_dim-1)
        gradient_h_gamma = np.zeros([self.state_dim-1, self.sym_dim])

        # Barrier function
        for k in range(self.state_dim-1):
            residues = np.sqrt( np.array([ (Z[:,k].T @ v0)**2, (Z[:,k+1].T @ v0)**2 ]) )
            max_index = np.argmax(residues)
            residue = residues[max_index]
            delta_lambda = pencil_eig[k+1] - pencil_eig[k]
            h1 = (residue**2)/(delta_lambda**2)
            h_gamma[k] = np.log(h1) - self.f_params_dict["epsilon"]
            
            # Barrier function gradient
            C = 2*residue/(delta_lambda**2)/h1
            for i in range(self.sym_dim):
                term1 = ( Z[:,max_index].T @ partial_Hv[i] @ ( p0 - x0 ) )
                term2 = (residue/delta_lambda)*( Z[:,k+1].T @ partial_Hv[i] @ Z[:,k+1] - Z[:,k].T @ partial_Hv[i] @ Z[:,k] )
                gradient_h_gamma[k,i] = C*(term1 - term2)

        # print("h_gamma = " + str(h_gamma))
        # print("grad h_gamma = " + str(gradient_h_gamma))

        return h_gamma, gradient_h_gamma

    # def get_mode(self):
    #     '''
    #     Computes the operation mode for a general n-th order system (for now, only for second order systems - determinant suffices).
    #     Positive for definite barrier Hessian, negative otherwise.
    #     '''
    #     Hh = self.cbf.get_hessian()
    #     return np.linalg.det(Hh)

    # def get_region(self):
    #     '''
    #     Computes the region function: positive when is necessary to compatibilize, negative otherwise.
    #     '''                
    #     state = self.plant.get_state()
    #     return - ( self.cbfs[0].evaluate_function(*state) - self.f_params_dict["compatibility_threshold"] )

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

        # Compute the pencil eigenvectors
        pencil_eigenvectors = np.zeros([self.state_dim,self.state_dim])
        for k in range(len(pencil_eig)):
            eig, Q = np.linalg.eig( pencil_eig[k]*Hh - Hv )
            for i in range(len(eig)):
                if np.abs(eig[i]) <= 0.000001:
                    normalization_const = 1/np.sqrt(Q[:,i].T @ Hh @ Q[:,i])
                    pencil_eigenvectors[:,k] = normalization_const * Q[:,i]
                    break

        # Assumption: Hv is invertible => detHv != 0
        detHv = np.linalg.det(Hv)

        # Computes the pencil characteristic polynomial and denominator of f(\lambda)
        pencil_det = np.real(np.prod(pencil_eig))
        pencil_char = ( detHv/pencil_det ) * np.real(np.polynomial.polynomial.polyfromroots(pencil_eig))

        self.pencil_dict["eigenvalues"] = pencil_eig[sorted_args]
        self.pencil_dict["eigenvectors"] = pencil_eigenvectors[:,sorted_args]
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
        Hv = self.clf.get_hessian()
        x0 = self.clf.get_critical()

        Hh = self.active_cbf.get_hessian()
        p0 = self.active_cbf.get_critical()

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

        residues, poles, k = signal.residue( np.flip(num_poly), np.flip(den_poly), tol=0.001, rtype='avg' )

        index = np.argwhere(np.real(residues) < 0.0000001)
        residues = np.real(np.delete(residues, index))

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
            "repeated_poles": repeated_poles,
            "residues": residues,
        }

    def compute_equilibrium(self):
        '''
        Compute equilibrium solutions and equilibrium points.
        '''
        p0 = self.active_cbf.get_critical()
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
        self.compute_active_cbf()
        if self.active_cbf:
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
        Hv, x0 = self.clf.get_hessian(), self.clf.get_critical(), 
        p0 = self.active_cbf.get_critical()
        v0 = Hv.dot( p0 - x0 )

        H = self.pencil_value( lambda_var )
        H_inv = np.linalg.inv(H)
        return H_inv.dot(v0)

    def pencil_value(self, lambda_var):
        '''
        This function returns the value of the matrix pencil H(lambda) = lambda Hh - Hv
        '''
        Hv = self.clf.get_hessian()
        Hh = self.active_cbf.get_hessian()
        return lambda_var*Hh - Hv

    # def f_values2(self, args):
    #     '''
    #     Returns the values of f.
    #     '''
    #     numpoints = len(args)
    #     fvalues = np.zeros(numpoints)
    #     poles = self.f_dict["poles"]
    #     residues = self.f_dict["residues"]
    #     for k in range(numpoints):
    #         for i in range(len(residues)):
    #             fvalues[k] = fvalues[k] + residues[i]/(( args[k] - poles[i] )**2)
    #     return fvalues

    # def f_values3(self, args):
    #     '''
    #     Returns the values of f.
    #     '''
    #     numpoints = len(args)
    #     fvalues = np.zeros(numpoints)
    #     Hh = self.cbf.get_hessian()
    #     for k in range(numpoints):
    #         v = self.v_values(args[k])
    #         fvalues[k] = v.T @ Hh @ v
    #     return fvalues

