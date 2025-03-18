import numpy as np
import scipy as sp
import warnings 

from copy import copy
from numpy.polynomial import Polynomial as Poly

from dynamic_systems import Integrator, LinearSystem, DriftLess, Unicycle
from functions import QuadraticLyapunov, QuadraticBarrier, param2H, H2param

from quadratic_program import QuadraticProgram
from controllers.compatibility import QFunction

class CompatibleQP():
    '''
    Class for the compatible QP controller.
    '''
    def __init__(self, 
                 plant: LinearSystem | DriftLess, 
                 clf: QuadraticLyapunov, 
                 cbfs: QuadraticBarrier=[], 
                 alpha = 1.0, beta = 1.0, p = 1.0, kappa = 1.0, dt = 0.001,
                 **kwargs):

        if not isinstance(plant, (LinearSystem, DriftLess) ):
            raise Exception("Compatible controller only implemented for LTI and driftless full-rank systems.")
        
        self.plant = plant

        if not isinstance(clf, QuadraticLyapunov):
            raise Exception("Compatible controller only implemented for quadratic CLFs.")

        # Checks if CLF satisfies the CLF condition BEFORE creating the Q-functions
        self.clf = clf
        self.clf.satisfy_clf(A=self.plant.A)
        self.ref_clf = QuadraticLyapunov(hessian=self.clf.H, center=self.clf.center)

        if not isinstance(cbfs, list):
            raise Exception("Must pass a list of CBFs.")

        for cbf in cbfs:
            if not isinstance(cbf, QuadraticBarrier):
                raise Exception("Compatible controller only implemented for quadratic CBFs.")
        
        self.cbfs = cbfs
        self.num_cbfs = len(self.cbfs)

        if self.num_cbfs > 0:
            if clf._dim != cbfs[0]._dim: 
                raise Exception("CLF and CBF dimensions are not equal.")
            if np.any([ cbfs[0]._dim != cbf._dim for cbf in cbfs ]):
                raise Exception("CBF dimensions are not equal.")

        self.verbose = False
        self.active = True
        self.compatibilization = True
        if "active" in kwargs.keys():
            self.active = kwargs["active"]
        if "compatibilization" in kwargs.keys():
            self.compatibilization = kwargs["compatibilization"]
        if "verbose" in kwargs.keys():
            self.verbose = kwargs["verbose"]
        
        self.n, self.m = self.plant.n, self.plant.m
        self.sym_dim = int(( self.n * ( self.n + 1 ) )/2)

        # Initialize rate CLF parameters
        # self.Vpi = 0.0
        # self.gradient_Vpi = np.zeros(self.sym_dim)

        # Parameters for the inner and outer QPs
        self.alpha, self.beta, self.p = alpha, beta, p
        self.kappa = kappa
        self.gamma_poly = Poly([0.0, self.p])               # gamma_poly is a linear function; TO DO: generalize to class K

        # Parameters for the inner QP controller (QP1)
        self.QP1_dim = self.m + 1
        P1 = np.eye(self.QP1_dim)
        P1[-1,-1] = self.p
        q1 = np.zeros(self.QP1_dim)
        self.QP1 = QuadraticProgram(P=P1,q=q1)
        self.QP1_sol = np.zeros(self.QP1_dim)

        # Defines CLF shape dynamics 
        param = H2param( self.clf.H )
        self.clf_dynamics = Integrator(n=len(param), state=param)

        # Parameters for the outer QP controller (QP2)
        # self.QP2_dim = self.sym_dim + 1
        # P2 = np.eye(self.QP2_dim)
        # P2[-1,-1] = self.p[1]
        # q2 = np.zeros(self.QP2_dim)
        # self.QP2 = QuadraticProgram(P=P2,q=q2)
        # self.QP2_sol = np.zeros(self.QP2_dim)

        # Variable initialization
        self.ctrl_dt = dt
        self.V = 0.0
        self.u = np.zeros(self.m)
        self.u_v = np.zeros(self.sym_dim)

        if type(self.plant) == Unicycle: 
            self.radius = 1.0

        # Create Q-function list
        self.Qfunctions: list[QFunction] = [ QFunction(self.plant, self.ref_clf, cbf, self.p) for cbf in self.cbfs ]
        self.compatible_Hv = [ None for _ in self.Qfunctions ]

        self.active_index = None
        self.compatibilized = False

        if self.active and self.compatibilization:
            self.compatibilize(verbose=self.verbose)

        self.get_equilibria()

    def compatibilize(self, **kwargs):
        ''' 
        Tries to compatibilize all QFunctions.
        Finds N compatible CLF Hessians, one for each CBF.
        '''
        verbose = self.verbose
        if "verbose" in kwargs.keys():
            verbose = kwargs["verbose"]

        if verbose:
            print(f"Initializing compatibilization of {self.num_cbfs} CLF-CBF pairs...")

        for k, qfun in enumerate(self.Qfunctions):

            # Only compatibilize if Q-function is not initially already compatible.
            if qfun.is_compatible():
                Hv = copy(qfun.Hv)
                self.compatible_Hv[k] = Hv
                if verbose:
                    print(f"{k+1}-th CLF-CBF pair is already compatible with \nHv{k+1} = \n{Hv}\n, moving on...")
                continue

            if verbose:
                print(f"Initializing compatibilization of the {k+1}-th CLF-CBF pair...")

            # Compatibilize each Q-function and store resulting shapes
            results = qfun.compatibilize(verbose=self.verbose)

            if np.any( results["compatibility"] < -1e-2 ) or results["monotonicity"] < -1e-2:
                raise Exception(f"Compatibility failed.")

            Hv = results["Hv"]
            self.compatible_Hv[k] = Hv
            if verbose:
                print(f"{k+1}-th CLF-CBF pair sucessfully compatibilized!\nHv{k+1} = \n{Hv}")

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
        if not self.active:
            return np.zeros(len(self.u))

        # Configure QP1 control constraints and generate new control input
        A, b = self.get_active_clf_constraint()
        b = np.array([b])

        for cbf in self.cbfs:
            a_cbf, b_cbf = self.get_cbf_constraint(cbf)
            A = np.vstack( [ A, a_cbf ])
            b = np.hstack( [ b, b_cbf ])

        self.QP1.set_inequality_constraints(A, b)
        QP1_sol = self.QP1.get_solution()
        self.u = QP1_sol[0:self.m]

        # Configure QP1 control constraints for computation 
        # of the activation regions for each barrier
        A2, b2 = self.get_ref_clf_constraint()
        b2 = np.array([b2])
        for cbf in self.cbfs:
            a_cbf, b_cbf = self.get_cbf_constraint(cbf)
            A2 = np.vstack( [ A2, a_cbf ])
            b2 = np.hstack( [ b2, b_cbf ])
        self.QP1.set_inequality_constraints(A2, b2)
        self.QP1_sol = self.QP1.get_solution()

        # active_index = self.active_cbf_index()
        # print(f"Active CBF = {active_index}")

        return self.u

    def get_clf_control(self):
        ''' Computes the solution of the QP for CLF shape '''

        if not self.compatibilization:
            return np.zeros(len(self.u_v))

        # a_rate, b_rate = self.get_rate_constraint()
        # A_outer = a_rate
        # b_outer = np.array([ b_rate ])

        # self.QP2.set_inequality_constraints(A_outer, b_outer)

        # # Solve outer loop QP
        # QP2_sol = self.QP2.get_solution()
        # self.u_v = QP2_sol[0:self.sym_dim]

        # If some CBF is active, get the corresponding compatible shape as reference;
        # Otherwise, use original CLF instead
        ref_Hv = self.ref_clf.H
        if self.compatibilized:
            active_index = self.active_cbf_index()
            if active_index >= 0:
                ref_Hv = self.compatible_Hv[active_index]

        # Get current CLF shape state and constructs the proportional controller 
        pi = self.clf_dynamics.get_state()
        self.u_v = - self.kappa * ( pi - H2param(ref_Hv) ) 

        return self.u_v

    def get_plant_state(self):
        ''' Returns the current plant state x, f(x) and g(x) '''

        # Affine plant dynamics
        if type(self.plant) == Unicycle:
            x = self.plant.get_state()[:2]
            f = self.plant.f(x)[:2]
            g = self.plant.g(x)[:2,:]
        else:
            x = self.plant.get_state()
            f = self.plant.f(x)
            g = self.plant.g(x)

        print(f"State = {x}")

        return x, f, g

    def get_active_clf_constraint(self):
        ''' Returns the active Lyapunov constraint. '''

        # Get plant state
        x, f, g = self.get_plant_state()

        # Lyapunov function and gradient
        self.V, nablaV, Hv = self.clf.inverse_gamma_transform(x, self.gamma_poly)

        # Lie derivatives
        LfV = nablaV.dot(f)
        LgV = g.T.dot(nablaV)

        # Gradient w.r.t. pi
        partial_Hv = self.clf.partial_Hv()
        self.nablaV_pi = np.zeros(self.sym_dim)
        delta_x = ( x - self.clf.center ).reshape(self.n,1)
        for k in range(self.sym_dim):
            self.nablaV_pi[k] = 0.5 * ( delta_x.T @ partial_Hv[k] @ delta_x )[0,0]

        # CLF constraint for the first QP
        a_clf = np.hstack([ LgV, -1.0 ])
        b_clf = - self.alpha * self.V - LfV
        # b_clf += - self.nablaV_pi.T @ self.u_v

        return a_clf, b_clf

    def get_cbf_constraint(self, cbf: QuadraticBarrier):
        ''' Returns the CBF constraint. '''

        # Get plant state
        x, f, g = self.get_plant_state()

        # Barrier function and gradient
        h, nablah = cbf(x), cbf.gradient(x)

        # Lie derivatives
        Lfh = nablah.dot(f)
        Lgh = g.T.dot(nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack([ Lgh, 0.0 ])
        b_cbf = self.beta * h + Lfh

        return a_cbf, b_cbf

    def get_ref_clf_constraint(self):
        ''' Returns the reference Lyapunov constraint. '''

        # Get plant state
        x, f, g = self.get_plant_state()

        # Lyapunov function and gradient
        ref_V, ref_nablaV, ref_Hv = self.ref_clf.inverse_gamma_transform(x, self.gamma_poly)

        # Lie derivatives
        LfV = ref_nablaV.dot(f)
        LgV = g.T.dot(ref_nablaV)

        # CLF constraint for the first QP
        a_ref_clf = np.hstack([ LgV, -1.0 ])
        b_ref_clf = - self.alpha * ref_V - LfV

        return a_ref_clf, b_ref_clf

    def get_rate_constraint(self):
        ''' DEPRECATED. Sets the Lyapunov rate constraint. '''

        ref_Hv = self.ref_clf.H

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
        a_clf_pi = np.hstack( [ self.gradient_Vpi, 0.0 ])
        b_clf_pi = -self.alpha * self.Vpi

        return a_clf_pi, b_clf_pi

    def update_clf_dynamics(self, shape_ctrl):
        '''
        Integrates the dynamic system for the CLF Hessian matrix and updates the active CLF-CBF pair.
        '''
        self.clf_dynamics.set_control(shape_ctrl)
        self.clf_dynamics.actuate(self.ctrl_dt)
        param = self.clf_dynamics.get_state()

        Hv = param2H(param)
        self.clf.set_params(hessian=Hv)

    def active_cbf_index(self):
        '''
        Returns the index of the current active CBF, if only one CBF is active.
        Returns -1 otherwise.
        '''
        cbf_constraints = []
        for cbf in self.cbfs:
            a_cbf, b_cbf = self.get_cbf_constraint(cbf)
            cbf_constraints.append( -a_cbf @ self.QP1_sol + b_cbf )

        arr = np.array(cbf_constraints) <= np.array([ 1e-5 for _ in range(len(self.cbfs)) ])

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
        
        # return 0
        return -1

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