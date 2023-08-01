import numpy as np
from controllers import CLFCBFPair
from dynamic_systems import Unicycle
from quadratic_program import QuadraticProgram
from common import sat


class NominalQP():
    '''
    Class for the nominal QP controller.
    '''
    def __init__(self, plant, clf, cbfs, alpha = 1.0, beta = 1.0, p = 10.0, dt = 0.001):

        # Dimensions and system model initialization
        self.plant = plant
        self.clf, self.cbfs = clf, cbfs
        
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
        self.clf_cbf_pairs = []
        self.equilibrium_points = np.zeros([0,clf_dim])
        for cbf in cbfs:
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
        QP_sol = self.QP.get_solution()
        control = QP_sol[0:self.control_dim,]

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


class NominalPF(NominalQP):
    '''
    Class for the nominal path following QP-based controller.
    '''
    def __init__(self, path, plant, clf, cbfs, alpha = 1.0, beta = 1.0, p = 10.0, dt = 0.001):
        super().__init__(plant, clf, cbfs, alpha = 1.0, beta = 1.0, p = 10.0, dt = 0.001)
        self.path = path
        self.path_speed = 4.0
        self.toggle_threshold = 1.0
        self.kappa = 1.0
        self.dgamma = 0.0

    def get_control(self):
        '''
        Modifies get_control() function for path following functionality.
        '''
        control = super().get_control()

        # Updates path dynamics
        gamma = self.path.get_path_state()
        # xd = self.path.get_path_point( gamma )
        dxd = self.path.get_path_gradient( gamma )

        # tilde_x = self.plant.get_state() - xd
        # if np.linalg.norm(tilde_x) >= self.toggle_threshold:
        #     eta_e = -tilde_x.dot( dxd )
        #     self.dgamma = - self.kappa * sat(eta_e, limits=[-10.0,10.0])
        # else:
        #     self.dgamma = self.path_speed/np.linalg.norm( dxd )

        self.dgamma = self.path_speed/np.linalg.norm( dxd )
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