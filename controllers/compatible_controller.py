import numpy as np

from quadratic_program import QuadraticProgram
from functions import QuadraticLyapunov, QuadraticBarrier
from dynamic_systems import LinearSystem, DriftLess, Unicycle
from controllers.compatibility import MatrixPencil, QFunction

class CompatibleQP():
    '''
    Class for the compatible QP controller.
    '''
    def __init__(self, 
                 plant: LinearSystem | DriftLess, 
                 clf: QuadraticLyapunov, 
                 cbfs: QuadraticBarrier, 
                 alpha = [1.0, 1.0], beta = [1.0, 1.0], p = [10.0, 10.0], dt = 0.001):

        if not isinstance(plant, (LinearSystem, DriftLess) ):
            raise Exception("Compatible controller only implemented for LTI and driftless full-rank systems.")
        
        if not isinstance(clf, QuadraticLyapunov):
            raise Exception("Compatible controller only implemented for quadratic CLFs.")

        if not isinstance(cbfs, list) or len(cbfs) == 0:
            raise Exception("Must pass a non-empty list of CBFs.")

        for cbf in cbfs:
            if not isinstance(cbf, QuadraticBarrier):
                raise Exception("Compatible controller only implemented for quadratic CBFs.")

        self.plant = plant
        self.clf = clf
        self.cbfs = cbfs
        self.num_cbfs = len(self.cbfs)

        self.n, self.m = self.plant.n, self.plant.m
        self.Hv = self.clf.H

        if clf._dim != cbfs[0]._dim: 
            raise Exception("CLF and CBF dimensions are not equal.")
        if np.any([ cbfs[0]._dim != cbf._dim for cbf in cbfs ]):
            raise Exception("CBF dimensions are not equal.")

        self.Qfunctions: list[QFunction] = [ None for cbf in cbfs ]
        self.compatible_Hv: list[np.ndarray] = [ None for cbf in cbfs ]

        self.state_dim = clf._dim
        self.control_dim = self.plant.m
        self.sym_dim = int(( self.state_dim * ( self.state_dim + 1 ) )/2)
        self.skewsym_dim = int(( self.state_dim * ( self.state_dim - 1 ) )/2)

        # Initialize rate CLF
        self.Vpi = 0.0
        self.gradient_Vpi = np.zeros(self.sym_dim)

        self.mode_log = []                   # mode = 1 for compatibility, mode = 0 for rate
        self.pencil_dict = {}
        self.f_params_dict = {
            "epsilon": 0.1,
            "min_CLF_eigenvalue": 0.2
        }
        self.clf.set_params(epsilon=self.f_params_dict["min_CLF_eigenvalue"])

        # Create CLF-CBF pairs
        # self.active_pair = None
        # self.clf_cbf_pairs = []
        # self.ref_clf_cbf_pairs = []
        # self.equilibrium_points = np.zeros([0,self.state_dim])
        # for cbf in cbfs:
        #     self.clf_cbf_pairs.append( CLFCBFPair(self.clf, cbf) )
        #     ref_clf_cbf_pair = CLFCBFPair(self.ref_clf, cbf)
        #     self.ref_clf_cbf_pairs.append( ref_clf_cbf_pair )
        #     self.equilibrium_points = np.vstack([ self.equilibrium_points, ref_clf_cbf_pair.equilibrium_points.T ])
        # self.h_gamma = np.zeros(self.state_dim-1)

        # Parameters for the inner and outer QPs
        self.alpha, self.beta, self.p = alpha, beta, p

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

        if type(self.plant) == Unicycle:
            self.radius = 1.0

        self.createQfunctions()

    def createQfunctions(self):
        ''' Computes Qfunctions for each CBF. '''

        for k, cbf in enumerate(self.cbfs):
            if isinstance(self.plant, LinearSystem):
                A, B = self.plant._A, self.plant._B
                G = B @ B.T
                Hh = cbf.H
                M, N = G @ Hh, self.p[0] * G @ self.Hv - A
            if isinstance(self.plant, DriftLess):
                Hh = cbf.H
                M, N = Hh, self.p[0] @ self.Hv

            w = N @ ( cbf.center - self.clf.center )
            pencil = MatrixPencil(M, N)
            self.Qfunctions[k] = QFunction(pencil, Hh, w)

    def compatibilize(self):
        ''' 
        Tries to compatibilize all QFunctions.
        Finds N CLF Hessians, one for each CBF, resulting in no 
        boundary equilibrium points.
        '''
        for k, qfun in enumerate(self.Qfunctions):

            if isinstance(self.plant, LinearSystem):
                A, B = self.plant._A, self.plant._B
            if isinstance(self.plant, DriftLess):
                A, B = np.zeros((self.n, self.n)), np.eye(self.n)

            results = qfun.compatibilize(A, B, self.clf.param2Hv, self.clf.Hv2param, self.Hv, self.p[0] )
            # if results["compatibility"] < 0:
            #     raise Exception(f"Compatibility failed.")
            # else:
            print(f"Compatibilization results = {results}")
            self.compatible_Hv[k] = results["Hv"]

        return self.compatible_Hv

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
        if self.active_pair:
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
        if type(self.plant) == Unicycle:
            f = self.plant.get_f()[:2]
            g = self.plant.get_g()[:2,:]
            state = self.plant.get_state()[:2]
        else:
            f = self.plant.get_f()
            g = self.plant.get_g()
            state = self.plant.get_state()

        # Lyapunov function and gradient
        self.V = self.clf(state)
        nablaV = self.clf(state)

        # Lie derivatives
        LfV = nablaV.dot(f)
        LgV = g.T.dot(nablaV)

        # Gradient w.r.t. pi
        # partial_Hv = self.clf.get_partial_Hv()
        # self.nablaV_pi = np.zeros(self.sym_dim)
        # delta_x = ( state - self.clf.get_critical() ).reshape(self.state_dim,1)
        # for k in range(self.sym_dim):
        #     self.nablaV_pi[k] = 0.5 * ( delta_x.T @ partial_Hv[k] @ delta_x )[0,0]

        # CLF constraint for the first QP
        a_clf = np.hstack([ LgV, -1.0 ])
        b_clf = - self.alpha[0]*self.V - LfV

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
        h = cbf(state)
        nablah = cbf(state)

        Lfh = nablah.dot(f)
        Lgh = g.T.dot(nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack([ Lgh, 0.0 ])
        b_cbf = self.beta[0] * h + Lfh

        return a_cbf, b_cbf

    def update_clf_dynamics(self, piv_ctrl):
        '''
        Integrates the dynamic system for the CLF Hessian matrix and updates the active CLF-CBF pair.
        '''
        self.clf.update(piv_ctrl, self.ctrl_dt)

        index = self.active_cbf_index()
        if index >= 0:
            # self.active_cbf = self.cbfs[index]
            self.active_pair = self.clf_cbf_pairs[index]
            self.active_pair.update( clf = self.clf )
        else:
            # self.active_cbf = None
            self.active_pair = None        
        
        # print(self.active_pair)

    # def update_cbf_dynamics(self, pih_ctrl):
    #     '''
    #     Integrates the dynamic system for the CBF Hessian matrix.
    #     '''
    #     self.active_cbf.update(pih_ctrl, self.ctrl_dt)
    #     self.active_pair.update( cbf = self.cbf )

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
        partial_Hv = self.clf.get_partial_Hv()

        self.Vpi = 0.5 * np.trace( deltaHv @ deltaHv )
        for k in range(self.sym_dim):
            self.gradient_Vpi[k] = np.trace( deltaHv @ partial_Hv[k] )

        # Sets rate constraint
        a_clf_pi = np.hstack( [ self.gradient_Vpi, -1.0 ])
        b_clf_pi = -self.alpha[1] * self.Vpi

        return a_clf_pi, b_clf_pi

    def get_eigenvector_constraints(self):
        '''
        Sets the constraint for fixing the pencil eigenvectors.
        '''
        JacobianV = np.zeros([self.skewsym_dim, self.sym_dim])
        Z = self.active_pair.pencil.eigenvectors
        partial_Hv = self.active_pair.clf.get_partial_Hv()

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
        self.h_gamma, Lg_h_gamma, Lf_h_gamma = self.compatibility_barrier()
        a_cbf_pi = -np.hstack([ Lg_h_gamma, np.zeros([self.state_dim-1, 1]) ])
        b_cbf_pi = self.beta[1]*self.h_gamma + Lf_h_gamma

        return a_cbf_pi, b_cbf_pi

    def compatibility_barrier(self):
        '''
        Computes compatibility barrier constraint, for keeping the critical values of f above 1.
        '''
        Hv = self.active_pair.clf.get_hessian()
        partial_Hv = self.active_pair.clf.get_partial_Hv()
        pencil_eig = self.active_pair.pencil.eigenvalues
        Z = self.active_pair.pencil.eigenvectors

        # Compatibility barrier
        h_gamma = np.zeros(self.state_dim-1)
        Lg_h_gamma = np.zeros([self.state_dim-1, self.sym_dim])
        Lf_h_gamma = np.zeros(self.state_dim-1)

        # Barrier function
        for k in range(self.state_dim-1):
            residues = np.sqrt( np.array([ (Z[:,k].T @ self.active_pair.v0)**2, (Z[:,k+1].T @ self.active_pair.v0)**2 ]) )
            max_index = np.argmax(residues)
            residue = residues[max_index]
            delta_lambda = pencil_eig[k+1] - pencil_eig[k]
            h1 = (residue**2)/(delta_lambda**2)
            h_gamma[k] = np.log(h1) - self.f_params_dict["epsilon"]
            
            # Barrier function gradient
            # C = 2*residue/(delta_lambda**2)/h1
            C = 2/residue
            for i in range(self.sym_dim):
                term1 = ( Z[:,max_index].T @ partial_Hv[i] @ ( self.active_pair.p0 - self.active_pair.x0 ) )
                term2 = (residue/delta_lambda)*( Z[:,k+1].T @ partial_Hv[i] @ Z[:,k+1] - Z[:,k].T @ partial_Hv[i] @ Z[:,k] )
                Lg_h_gamma[k,i] = C*(term1 - term2)

            dx0 = self.active_pair.clf.get_critical_derivative()
            dp0 = self.active_pair.cbf.get_critical_derivative()
            Lf_h_gamma[k] = C * Z[:,max_index].T @ Hv @ ( dp0 - dx0 )

        return h_gamma, Lg_h_gamma, Lf_h_gamma
    

class CompatiblePF(CompatibleQP):
    '''
    Class for the compatible path following QP-based controller.
    '''
    def __init__(self, path, plant, clf, ref_clf, cbfs, alpha = [1.0, 1.0], beta = [1.0, 1.0], p = [1.0, 1.0], dt = 0.001):
        super().__init__(plant, clf, ref_clf, cbfs, alpha = alpha, beta = beta, p = p, dt = dt)
        self.path = path
        self.path_speed = 4.0
        self.toggle_threshold = 1.0
        self.kappa = 1.0
        self.dgamma = 0.0

        self.evolving = True
        self.in_unsafe_region = False

    def get_control(self):
        '''
        Modifies get_control() function for path following functionality 
        '''
        # Updates path state
        gamma = self.path.get_path_state()
        xd = self.path.get_path_point( gamma )
        dxd = self.path.get_path_gradient( gamma )

        if self.active_pair != None:
            h_at_xd = self.active_pair.cbf.evaluate_function(*xd)[0]

            if h_at_xd < 0:
                self.in_unsafe_region = True

        if self.in_unsafe_region and self.active_pair == None:
            self.evolving = False
            self.in_unsafe_region = False

        if np.linalg.norm(self.plant.get_state() - xd) <= self.toggle_threshold:
            self.evolving = True

        self.dgamma = 0.0
        if self.evolving:
            self.dgamma = self.path_speed/np.linalg.norm( dxd )

        # self.dgamma = self.path_speed/np.linalg.norm( dxd )
        # tilde_x = self.plant.get_state() - xd
        # if np.linalg.norm(tilde_x) >= self.toggle_threshold:
        #     eta_e = -tilde_x.dot( dxd )
        #     self.dgamma = - self.kappa * sat(eta_e, limits=[-10.0,10.0])

        self.path.update(self.dgamma, self.ctrl_dt)

        # Updates the clf with new critical point and velocity
        gamma = self.path.get_path_state()
        xd = self.path.get_path_point( gamma )
        self.clf.set_critical(  xd )
        self.clf.set_critical_derivative( dxd * self.dgamma )

        # Computes the PF control
        control = super().get_control()

        return control

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
        LfV = nablaV @ ( f - self.clf.get_critical_derivative() )
        LgV = g.T.dot( nablaV )

        # CLF constraint for the first QP
        a_clf = np.hstack([ LgV, -1.0 ])
        b_clf = - self.alpha[0]*self.V - LfV

        return a_clf, b_clf