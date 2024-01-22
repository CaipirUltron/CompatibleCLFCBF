import numpy as np
import cvxpy as cp

from dynamic_systems import Unicycle
from quadratic_program import QuadraticProgram
from controllers.compatibility import CLFCBFPair
from controllers.equilibrium_algorithms import equilibrium_field, compute_equilibria, S

class NominalQP():
    '''
    Class for the nominal QP controller.
    '''
    def __init__(self, plant, clf, cbf, alpha = 1.0, beta = 1.0, p = 1.0, dt = 0.001):

        # Dimensions and system model initialization
        self.plant = plant
        self.clf, self.cbf = clf, cbf

        if self.clf.kernel != self.cbf.kernel:
            raise Exception("CLF and CBF must be based on the same kernel.")
        self.kernel = self.clf.kernel
        self.A_list = self.kernel.get_A_matrices()

        if self.clf._dim != self.cbf._dim:
            raise Exception("CLF and CBF dimensions are not equal.")

        self.state_dim = self.clf._dim
        self.control_dim = self.plant.m
        self.kernel_dim = self.kernel.kernel_dim

        # QP parameters
        self.p, self.alpha, self.beta = p, alpha, beta
        self.eq_params = {"slack_gain": self.p, "clf_gain": self.alpha}
        self.QP_dim = self.control_dim + 1
        P = np.eye(self.QP_dim)
        P[self.control_dim,self.control_dim] = self.p
        q = np.zeros(self.QP_dim)
        self.QP = QuadraticProgram(P=P, q=q)
        self.QP_sol = np.zeros(self.QP_dim)

        self.ctrl_dt = dt

        self.min_curvature = 0.5
        self.eq_dt = 0.1
        self.invariants = []
        self.equilibria = []
        self.has_pivot = False
        self.needs_update = False
        
        self.updated_timer = False
        self.timer = 0.0
        self.last_updated_by = None
        self.last_eq_t = 0.0

        '''
        Setup cvxpy problem for compatibility
        '''
        self.Pnom = cp.Parameter((self.kernel_dim,self.kernel_dim), symmetric=True)
        self.Pnew = cp.Variable((self.kernel_dim,self.kernel_dim), symmetric=True)
        self.l_pivot = cp.Variable()

        self.objective = cp.Minimize( cp.norm( self.Pnew - self.Pnom ) )
        self.constraints = { "psd": self.Pnew >> 0, "pivot": [], "gradient": [] }

    def get_control(self):
        '''
        Computes the QP control.
        '''
        # Gets CLF and CBF constraints
        a_clf, b_clf = self.get_clf_constraint()
        a_cbf, b_cbf = self.get_cbf_constraint()

        A = np.vstack([ a_clf, a_cbf ])
        b = np.hstack([ b_clf, b_cbf ])

        # Solve QP
        self.QP.set_inequality_constraints(A, b)
        self.QP_sol = self.QP.get_solution()
        control = self.QP_sol[0:self.control_dim,]

        elapsed_time = self.timer - self.last_eq_t
        if elapsed_time > self.eq_dt:
            x = self.plant.get_state().tolist()
            sol = compute_equilibria(self.plant, self.clf, self.cbf, self.eq_params, init_x=x)
            if sol["x"] != None: 
                self.add_equilibrium(sol)
                if not self.has_pivot: self.set_pivot(sol)

            self.garbage_collector()
            self.last_eq_t = self.timer

            if self.needs_update:
                self.update_clf()

        return control

    def update_clf(self):
        '''
        Updates the CLF for compatibility
        '''
        constraints = [ self.constraints["psd"] ]
        constraints += self.constraints["pivot"]
        problem = cp.Problem(self.objective, constraints)
        
        try:
            self.Pnom.value = self.clf.P
            problem.solve()
        except Exception as error:
            print("New CLF cannot be computed. Error: " + str(error))

        print("CLF updated. Optimization status exit as \"" + str(problem.status) + "\".")
        print("Optimal value is %s." % problem.value)

        if "optimal" in problem.status:
            self.clf.set_param(P=self.Pnew.value)

        self.needs_update = False

    def garbage_collector(self):
        '''
        Verifies equilibrium conditions for all equilibrium points, removing from the list the garbage ones
        '''
        for eq_sol in self.equilibria:
            x_e = eq_sol["x"]
            l_e = eq_sol["lambda"]
            m = self.kernel.function(x_e)
            V = self.clf.function(x_e)
            field = equilibrium_field(m, l_e, self.clf.P, V, self.plant, self.clf, self.cbf, self.eq_params)
            if np.linalg.norm(field) > 1e-6:
                del eq_sol

    def add_equilibrium(self, new_eq):
        '''
        Adds a new equilibrium point to the list.
        '''
        # If point is already in the list, pass
        for eq in self.equilibria:
            if np.linalg.norm(np.array(new_eq["x"]) - np.array(eq["x"])) < 1e-03:
                return
        self.equilibria.append(new_eq)

        # if len(self.equilibria) > 0:
        #     Pnew = closest_compatible(self.plant, self.clf, self.cbf, [self.equilibria[-1]], slack_gain=self.p, clf_gain=self.alpha)
        #     self.clf.set_param(P=Pnew)

    def equilibrium_constr(self, sol):
        '''
        Returns the cvxpy constraints to keep a given equilibrium sol as a valid equilibrium.
        '''
        x_e = sol["x"]
        m = self.kernel.function(x_e)
        V = self.clf.function(x_e)
        return [ equilibrium_field(m, self.l_pivot, self.Pnew, V, self.plant, self.clf, self.cbf, self.eq_params) == 0, 
                  V - 0.5 * m.T @ self.Pnew @ m == 0, 
                  self.l_pivot >= 0 ]

    def curvature_constr(self, sol, **kwargs):
        '''
        Returns the cvxpy curvature constraint for an equilibrium point sol
        '''
        c_lim = 1.0
        for key in kwargs.keys():
            aux_key = key.lower()
            if aux_key == "c_lim":
                c_lim = kwargs[key]
                # continue

        x_e = sol["x"]
        n = self.state_dim

        M = np.random.rand(n,n)
        nablah_e = self.cbf.gradient(x_e)
        normal = nablah_e / np.linalg.norm(nablah_e)

        M[:,0] = normal
        while np.abs( np.linalg.det(M) ) <= 1e-10:
            M = np.random.rand(n,n)
            M[:,0] = normal
        Rot, _ = np.linalg.qr(M)

        aux_M = np.vstack([ np.zeros(n-1), np.eye(n-1) ])
        S_matrix = S(x_e, self.l_pivot, self.Pnew, self.plant, self.clf, self.cbf, self.eq_params)

        return cp.lambda_min(aux_M.T @ Rot.T @ S_matrix @ Rot @ aux_M) >= c_lim

    def set_pivot(self, pivot_pt):
        '''
        Sets up the pivot equilibrium point.
        '''
        if pivot_pt not in self.equilibria:
            raise Exception("Pivot must be an equilibrium point.")
        
        self.constraints["pivot"] = []
        self.constraints["pivot"] += self.equilibrium_constr(pivot_pt)
        self.constraints["pivot"].append( self.curvature_constr(pivot_pt, clim=self.min_curvature) )
        self.has_pivot = True
        self.needs_update = True

    def get_clf_control(self):
        '''
        For now, the controller will not modify the CLF.
        '''
        return np.zeros(len(self.clf.param))

    def get_clf_constraint(self):
        '''
        Sets the Lyapunov constraint.
        '''
        # Affine plant dynamics
        state = self.plant.get_state()

        f = self.plant.get_f(state)
        g = self.plant.get_g(state)

        # Lyapunov function and gradient
        V = self.clf.function(state)
        nablaV = self.clf.gradient(state)

        # Lie derivatives
        LfV = nablaV.dot(f)
        LgV = g.T.dot(nablaV)

        # CLF contraint for the QP
        a_clf = np.hstack( [ LgV, -1.0 ])
        b_clf = -self.alpha * V - LfV

        return a_clf, b_clf

    def get_cbf_constraint(self):
        '''
        Sets the i-th barrier constraint.
        '''
        state = self.plant.get_state()

        f = self.plant.get_f(state)
        g = self.plant.get_g(state)

        # Barrier function and gradient
        h = self.cbf.function(state)
        nablah = self.cbf.gradient(state)

        Lfh = nablah.dot(f)
        Lgh = g.T @ nablah

        # CBF contraint for the QP
        a_cbf = -np.hstack( [ Lgh, 0.0 ])
        b_cbf = self.beta * h + Lfh

        return a_cbf, b_cbf

    def update_clf_dynamics(self, piv_ctrl):
        '''
        Integrates the dynamic system for the CLF Hessian matrix.
        '''
        self.clf.update(piv_ctrl, self.ctrl_dt)
        self.update_timer(self.update_clf_dynamics)

    def update_timer(self, method):
        '''
        Updates built-in timer
        '''
        if self.last_updated_by == method.__name__:
            self.updated_timer = False
        if not self.updated_timer:
            self.timer += self.ctrl_dt
            self.updated_timer = True
            self.last_updated_by = method.__name__

class NominalQuadraticQP():
    '''
    Class for the nominal QP controller.
    '''
    def __init__(self, plant, clf, cbfs, alpha = 1.0, beta = 1.0, p = 10.0, dt = 0.001):

        # Dimensions and system model initialization
        self.plant = plant
        self.clf = clf
        
        if type(cbfs) == list:
            self.cbfs = cbfs
        else:
            self.cbfs = []
            self.cbfs.append(cbfs)

        self.mode_log = None
        
        clf_dim = self.clf._dim
        cbf_dims = []

        for cbf in self.cbfs:
            cbf_dims.append( cbf._dim )

        if not all(dim == cbf_dims[0] for dim in cbf_dims): raise Exception("CBF dimensions are not equal.")
        if cbf_dims[0] != clf_dim: raise Exception("CLF and CBF dimensions are not equal.")

        self.state_dim = clf_dim
        self.control_dim = self.plant.m
        self.sym_dim = int(( self.state_dim * ( self.state_dim + 1 ) )/2)
        self.skewsym_dim = int(( self.state_dim * ( self.state_dim - 1 ) )/2)

        # Compute equilibrium points
        self.active_pair = None
        self.clf_cbf_pairs = []
        self.equilibrium_points = np.zeros([0,clf_dim])
        for cbf in self.cbfs:
            clf_cbf_pair = CLFCBFPair(self.clf, cbf)
            self.clf_cbf_pairs.append( clf_cbf_pair )
            self.equilibrium_points = np.vstack([ self.equilibrium_points, clf_cbf_pair.equilibrium_points.T ])

        # QP parameters
        self.p, self.alpha, self.beta = p, alpha, beta
        self.QP_dim = self.control_dim + 1
        P = np.eye(self.QP_dim)
        P[self.control_dim,self.control_dim] = self.p
        q = np.zeros(self.QP_dim)
        self.QP = QuadraticProgram(P=P, q=q)
        self.QP_sol = np.zeros(self.QP_dim)

        self.ctrl_dt = dt
        if type(self.plant) == Unicycle:
            self.radius = 1.0

    def get_control(self):
        '''
        Computes the QP control.
        '''
        A, b = self.get_clf_constraint()

        # Stacking the CLF and CBF constraints
        for cbf in self.cbfs:
            a_cbf, b_cbf = self.get_cbf_constraint(cbf)
            A = np.vstack( [ A, a_cbf ])
            b = np.hstack( [ b, b_cbf ])

        # Solve QP
        self.QP.set_inequality_constraints(A, b)
        self.QP_sol = self.QP.get_solution()
        control = self.QP_sol[0:self.control_dim,]

        return control

    def get_clf_control(self):
        '''
        This controller will not modify the CLF.
        '''
        return np.zeros(self.sym_dim)

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
        self.V = self.clf.evaluate_function(*state)[0]
        self.nablaV = self.clf.evaluate_gradient(*state)[0]
        
        # Lie derivatives
        self.LfV = self.nablaV.dot(f)
        self.LgV = g.T.dot(self.nablaV)

        # CLF contraint for the QP
        a_clf = np.hstack( [ self.LgV, -1.0 ])
        b_clf = -self.alpha * self.V - self.LfV

        return a_clf, b_clf

    def get_cbf_constraint(self, cbf):
        '''
        Sets the i-th barrier constraint.
        '''
        if type(self.plant) == Unicycle:
            state = self.plant.get_state()[:2]
            phi = self.plant.get_state()[2]
            robot_pose = ( state[0], state[1], phi )
            robot_center = self.plant.geometry.get_center(robot_pose)

            h, nablah, closest_pt, gamma_opt = cbf.barrier_set({"radius": self.radius, "center": robot_center, "orientation": phi})

            f = self.plant.get_f()[:2]
            g = np.array([[ np.cos(phi), -self.radius*np.sin(phi+gamma_opt) ],[ np.sin(phi), self.radius*np.cos(phi+gamma_opt) ]])

        else:
            f = self.plant.get_f()
            g = self.plant.get_g()
            state = self.plant.get_state()

            # Barrier function and gradient
            h = cbf.evaluate_function(*state)[0]
            nablah = cbf.evaluate_gradient(*state)[0]

        self.Lfh = nablah.dot(f)
        self.Lgh = g.T @ nablah

        # CBF contraint for the QP
        a_cbf = -np.hstack( [ self.Lgh, 0.0 ])
        b_cbf = self.beta * h + self.Lfh

        return a_cbf, b_cbf

    def update_clf_dynamics(self, piv_ctrl):
        '''
        Integrates the dynamic system for the CLF Hessian matrix.
        '''
        self.clf.update(piv_ctrl, self.ctrl_dt)

    def update_cbf_dynamics(self, cbf, pih_ctrl):
        '''
        Integrates the dynamic system for the CBF Hessian matrix.
        '''
        cbf.update(pih_ctrl, self.ctrl_dt)

class NominalPF(NominalQuadraticQP):
    '''
    Class for the nominal path following QP-based controller.
    '''
    def __init__(self, path, plant, clf, cbfs, alpha = 1.0, beta = 1.0, p = 10.0, dt = 0.001):
        super().__init__(plant, clf, cbfs, alpha = alpha, beta = beta, p = p, dt = dt)
        self.path = path
        self.path_speed = 4.0
        self.toggle_threshold = 1.0
        self.kappa = 1.0
        self.dgamma = 0.0

        self.evolving = True
        self.in_unsafe_region = False

    def active_cbf_index(self):
        '''
        Returns the index of the current active CBF, if only one CBF is active.
        Returns -1 otherwise.
        '''
        cbf_constraints = []
        for cbf in self.cbfs:
            a_cbf, b_cbf = self.get_cbf_constraint(cbf)
            cbf_constraints.append( -a_cbf @ self.QP_sol + b_cbf )

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

    def get_control(self):
        '''
        Modifies get_control() function for path following functionality.
        '''
        control = super().get_control()

        # Updates path dynamics
        gamma = self.path.get_path_state()
        xd = self.path.get_path_point( gamma )
        dxd = self.path.get_path_gradient( gamma )

        index = self.active_cbf_index()
        self.active_pair = None
        if index >= 0:
            self.active_pair = self.clf_cbf_pairs[index]               

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

        # Updates the clf with new critical point
        gamma = self.path.get_path_state()
        xd = self.path.get_path_point( gamma )
        self.clf.set_critical( xd )

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
        self.nablaV = self.clf.evaluate_gradient(*state)[0]
        
        # Lie derivatives
        gamma = self.path.get_path_state()
        dxd = self.path.get_path_gradient( gamma )
        self.LfV = self.nablaV @ ( f - dxd * self.dgamma )
        self.LgV = g.T.dot(self.nablaV)

        # CLF contraint for the QP
        a_clf = np.hstack( [ self.LgV, -1.0 ])
        b_clf = -self.alpha * self.V - self.LfV

        return a_clf, b_clf