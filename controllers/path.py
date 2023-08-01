import numpy as np
import scipy.interpolate as si
import scipy.optimize as opt
from dynamic_systems import Integrator


def Line(gamma, params):
    '''
    Method for a straight line path.
    '''
    starting_pt = params["start"]
    direction = params["direction"]
    direction = direction/np.linalg.norm(direction)

    xd = starting_pt + gamma*direction
    dxd = direction

    return xd, dxd


def Circle(gamma, params):
    '''
    Method for a circular path. 
    gamma = 0 corresponds to the top point on the circle.
    '''
    center_pt = params["center"]
    radius = params["radius"]

    xd = np.array(center_pt) + radius*np.array([ np.sin(gamma), np.cos(gamma) ])
    dxd = radius*np.array([ np.cos(gamma), -np.sin(gamma) ])

    return xd, dxd


class Path:
    '''
    This class implements path functionality.
    '''
    def __init__(self, function, params, init_path_state = [0.0]):

        self.path_func = function
        self.params = params
        self.system = Integrator(initial_state=init_path_state, initial_control = [0.0])

        print(self.system._control)

        self.gamma = init_path_state[0]
        self.dgamma = 0.0

        self.logs = {"gamma":[], "dgamma":[], "ddgamma":[]}

    def get_path_state(self):
        '''
        Function for getting the path states (gamma and dgamma)
        '''
        state = self.system.get_state()
        self.gamma = state[0]

        return self.gamma

    def get_path_control(self):
        '''
        Function for getting the path control (ddgamma)
        '''
        return self.dgamma

    def set_path_state(self, gamma):
        '''
        Sets (gamma, dgamma) to initial values.
        '''
        self.gamma = gamma
        self.system.set_state(self.gamma)

    def get_path_point(self, gamma):
        '''
        Returns the current virtual point position.
        '''
        xd, dxd = self.path_func(gamma, self.params)
        return xd

    def get_path_gradient(self, gamma):
        '''
        Returns the current virtual point gradient.
        '''
        xd, dxd = self.path_func(gamma, self.params)
        return dxd

    def update(self, gamma_ctrl, sample_time):
        '''
        Updates of the path.
        '''
        self.dgamma = gamma_ctrl

        self.system.set_control(gamma_ctrl)
        self.system.actuate(sample_time)

        self.get_path_state()
        self.log_path()

    def log_path(self):
        '''
        Logs the path variables.
        '''
        self.logs["gamma"] = self.system.state_log[0]
        self.logs["dgamma"] = self.system.control_log[0]

        return self.logs
    
    def draw_path(self, path_graph, plot_params):
        '''
        Draws the path
        '''
        numpoints = plot_params["numpoints"]
        path_length = plot_params["path_length"]

        xpath, ypath = [], []
        for k in range(numpoints):
            gamma = k*path_length/numpoints
            pos = self.get_path_point(gamma)
            xpath.append(pos[0])
            ypath.append(pos[1])
        path_graph.set_data(xpath, ypath)