import math
import numpy as np
from scipy.integrate import ode
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier


class SimulateDynamics:
    def __init__(self, plant, initial_state):
        
        # Simulation attributes
        self._dynamic_system = plant
        self.n = np.size(self._dynamic_system.state())
        self.m = np.size(self._dynamic_system.control())
        
        self._state = initial_state
        self._dstate = np.zeros(self.n)

        self.mODE = ode(self.dynamics).set_integrator('dopri5')
        self.mODE.set_initial_value(self._state)

        self.state_log = []
        for _ in range(0, self.n):
            self.state_log.append([])

        self.control_log = []
        for _ in range(0, self.m):
            self.control_log.append([])

    def send_control_inputs(self, control_input, dt):

        self.compute_dynamics(control_input)
        self._state = self.mODE.integrate(self.mODE.t+dt)

        for state_dim in range(0, self.n):
            self.state_log[state_dim].append(self._state[state_dim])

        for ctrl_dim in range(0, self.n):
            self.control_log[ctrl_dim].append(control_input[ctrl_dim])

        return self._state

    def compute_dynamics(self, control_input):
        dstate_tuple = self._dynamic_system(self._state, control_input)
        for i in range(self.n):
            self._dstate[i] = dstate_tuple[i]

    def dynamics(self, t):
        return self._dstate

    def state(self):
        return self._state

    @staticmethod
    def sat(input, limit=math.pi):
        return np.tanh(input*math.pi/limit)