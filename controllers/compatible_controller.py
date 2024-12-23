import numpy as np
from copy import copy

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
                 alpha = [1.0, 1.0], beta = [1.0, 1.0], p = [10.0, 10.0], dt = 0.001,
                 **kwargs):

        if not isinstance(plant, (LinearSystem, DriftLess) ):
            raise Exception("Compatible controller only implemented for LTI and driftless full-rank systems.")
        
        if not isinstance(clf, QuadraticLyapunov):
            raise Exception("Compatible controller only implemented for quadratic CLFs.")

        if not isinstance(cbfs, list) or len(cbfs) == 0:
            raise Exception("Must pass a non-empty list of CBFs.")

        for cbf in cbfs:
            if not isinstance(cbf, QuadraticBarrier):
                raise Exception("Compatible controller only implemented for quadratic CBFs.")

        if clf._dim != cbfs[0]._dim: 
            raise Exception("CLF and CBF dimensions are not equal.")
        if np.any([ cbfs[0]._dim != cbf._dim for cbf in cbfs ]):
            raise Exception("CBF dimensions are not equal.")

        self.plant = plant
        self.clf = clf
        self.cbfs = cbfs
        self.num_cbfs = len(self.cbfs)

        self.compatibilization = True
        if "compatibilization" in kwargs.keys():
            self.compatibilization = kwargs["compatibilization"]

        self.n, self.m = self.plant.n, self.plant.m
        self.sym_dim = int(( self.n * ( self.n + 1 ) )/2)

        # Initialize rate CLF parameters
        self.clf.set_params(epsilon=0.01)
        self.Hv_ref = copy(self.clf.H)
        self.Vpi = 0.0
        self.gradient_Vpi = np.zeros(self.sym_dim)

        # Parameters for the inner and outer QPs
        self.alpha, self.beta, self.p = alpha, beta, p

        # Parameters for the inner QP controller (QP1)
        self.QP1_dim = self.m + 1
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
        self.u = np.zeros(self.m)
        self.u_v = np.zeros(self.sym_dim)

        if type(self.plant) == Unicycle: 
            self.radius = 1.0

        self.Qfunctions: list[QFunction] = [ QFunction(self.plant, self.clf, cbf, self.p[0]) for cbf in self.cbfs ]
        self.compatible_Hv = [ None for _ in self.Qfunctions ]

        # Create CLF-CBF pairs
        self.active_index = None
        self.compatibilized = False

        if self.compatibilization:
            self.compatibilize(verbose=True)

        self.get_equilibria()

    def compatibilize(self, verbose=False):
        ''' 
        Tries to compatibilize all QFunctions.
        Finds N CLF Hessians, one for each CBF, resulting in no 
        boundary equilibrium points.
        '''
        for k, qfun in enumerate(self.Qfunctions):

            # Only compatibilize if Q-function is not initially already compatible.
            if qfun.is_compatible():
                self.compatible_Hv[k] = copy(self.clf.H)
                if verbose:
                    print(f"CLF is already compatible with the {k+1}-th CBF. Bypassing compatibilization...")
                continue

            if verbose:
                print(f"Initializing compatibilization of CLF with the {k+1}-th CBF...")

            # Compatibilize each Q-function and store resulting shapes
            results = qfun.compatibilize(verbose=verbose)
            if results["compatibility"] < -1e-2:
                raise Exception(f"Compatibility failed.")

            Hv = results["Hv"]
            self.compatible_Hv[k] = Hv
            if verbose:
                print(f"{k+1}-th compatibilization sucessfull with Hv = \n{Hv}")

        if verbose:
            print(f"Compatibilization sucessfull for all {self.num_cbfs} CLF-CBF pairs!")

        self.get_equilibria()
        self.compatibilized = True

        return self.compatible_Hv

    def get_cbf_equilibria(self, index):
        ''' Returns equilibrium points of CBF index '''
        if index < 0 or index >= self.num_cbfs:
            raise Exception("Invalid index.")
        return [ eq for eq in self.Qfunctions[index].equilibrium_sols ]

    def get_equilibria(self):
        ''' Returns equilibrium points for all Q-functions '''

        self.equilibrium_points = []
        for index in range(0, self.num_cbfs):
            self.equilibrium_points += self.get_cbf_equilibria(index)

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
        self.QP1.set_inequality_constraints(A, b)
        self.QP1_sol = self.QP1.get_solution()
        self.u = self.QP1_sol[0:self.m]

        return self.u

    def get_clf_control(self):
        '''
        Computes the solution of the outer QP.
        '''
        if not self.compatibilization:
            return np.zeros(len(self.u_v))

        a_rate, b_rate = self.get_rate_constraint()
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
        nablaV = self.clf.gradient(state)

        # Lie derivatives
        LfV = nablaV.dot(f)
        LgV = g.T.dot(nablaV)

        # Gradient w.r.t. pi
        partial_Hv = self.clf.partial_Hv()
        self.nablaV_pi = np.zeros(self.sym_dim)
        delta_x = ( state - self.clf.center ).reshape(self.n,1)
        for k in range(self.sym_dim):
            self.nablaV_pi[k] = 0.5 * ( delta_x.T @ partial_Hv[k] @ delta_x )[0,0]

        # CLF constraint for the first QP
        a_clf = np.hstack([ LgV, -1.0 ])
        b_clf = - self.alpha[0] * self.V - LfV 
        b_clf += - self.nablaV_pi.T @ self.u_v

        return a_clf, b_clf

    def get_cbf_constraint(self, cbf: QuadraticBarrier):
        '''
        Sets the barrier constraint.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()
        state = self.plant.get_state()

        # Barrier function and gradient
        h = cbf(state)
        nablah = cbf.gradient(state)

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

        # index = self.active_cbf_index()
        # if index >= 0:
        #     # self.active_cbf = self.cbfs[index]
        #     self.active_pair = self.clf_cbf_pairs[index]
        #     self.active_pair.update( clf = self.clf )
        # else:
        #     # self.active_cbf = None
        #     self.active_pair = None        
        
        # print(self.active_pair)

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
        ref_Hv = self.Hv_ref

        # If some CBF is active, get the corresponding compatible shape as reference;
        # Otherwise, use original CLF instead
        if self.compatibilized:
            active_index = self.active_cbf_index()
            if active_index >= 0:
                ref_Hv = self.compatible_Hv[active_index]

        deltaHv = self.clf.H - ref_Hv
        partial_Hv = self.clf.partial_Hv()

        self.Vpi = 0.5 * np.trace( deltaHv.T @ deltaHv )
        for k in range(self.sym_dim):
            self.gradient_Vpi[k] = np.trace( deltaHv @ partial_Hv[k] )

        # Sets rate constraint
        a_clf_pi = np.hstack( [ self.gradient_Vpi, -1.0 ])
        b_clf_pi = -self.alpha[1] * self.Vpi

        return a_clf_pi, b_clf_pi

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