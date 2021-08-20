import numpy as np
from abc import ABC, abstractmethod

class DynamicSystem(ABC):
    '''
    Abstract class for dynamic systems.
    '''
    def __init__(self, init_state):

        self.state_dim = 3
        self.control_dim = 2

        self._state = np.zeros(self.state_dim)
        self._control = np.zeros(self.control_dim)
    
    @abstractmethod
    def get_control(self):
        '''
        This method gets the last control input.
        '''
        return self._control

    @abstractmethod
    def state(self):
        '''
        This method gets the current system state.
        '''
        return self._state