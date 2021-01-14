import numpy as np
from scipy.integrate import ode
from dynamic_systems import NonlinearSystem, LinearSystem

class SimulateDynamics:
    def __init__(self, dimension, state0 = np.zeros(2)):
        
        # Simulation attributes
        self.n = len(state0)
        self.state = state0
        self.d_state = np.zeros(self.n)
        self.mODE = ode(self.dynamics).set_integrator('dopri5')
        self.mODE.set_initial_value(self.state)

    def send_control_inputs(self, control_input, dt):
        self.compute_dynamics(control_input)  
        new_state = self.mODE.integrate(self.mODE.t+dt)
        self.state = new_state

        return self.state

    def dynamics(self, control_input):
        
        self.dstate = compute_vector_field()
        return self.dstate

    def compute_vector_field(self, control_input):
        

        return

    def get_state(self):
        return self.state