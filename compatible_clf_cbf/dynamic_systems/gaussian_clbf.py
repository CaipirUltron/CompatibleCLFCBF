import numpy as np
from compatible_clf_cbf.dynamic_systems import Function


class Gaussian(Function):
    '''
    Class for gaussian function of the type N(x) = c exp( -0.5 (x-mu).T Sigma (x-mu) )
    '''
    def __init__(self, init_value=0.0, **kwargs):
        super().__init__(init_value)

        self.c = 0.0
        if self._dim > 1:
            self.mu = np.zeros(self._dim)
            self.Sigma = np.zeros([self._dim,self._dim])
        else:
            self.mu = 0.0
            self.Sigma = 0.0
        self.set_param(**kwargs)

    def set_param(self, **kwargs):
        '''
        Sets the quadratic function parameters.
        '''
        for key in kwargs:
            if key == "constant":
                self.c = kwargs[key]
            if key == "mean":
                self.mu = np.array(kwargs[key])
            if key == "shape":
                if np.shape(kwargs[key]) != ( self._dim, self._dim ):
                    raise Exception('Shape matrix must be the same dimension as the mean.')
                self.Sigma = np.array(kwargs[key])

    def function(self):
        '''
        Gaussian function.
        '''
        v = self._var - self.mu
        self._function = self.c * np.exp( -0.5 * v.T @ self.Sigma @ v )

    def gradient(self):
        '''
        Gradient of Gaussian function.
        '''
        v = self._var - self.mu
        self._gradient = - self.c * np.exp( -0.5 * v.T @ self.Sigma @ v ) * ( self.Sigma @ v )

    def hessian(self):
        '''
        Hessian of Gaussian function.
        '''
        v = self._var - self.mu
        self._hessian = - self.c * np.exp( -v.T @ self.Sigma @ v ) * ( self.Sigma - np.outer( self.Sigma @ v, self.Sigma @ v ) )


class CLBF(Function):
    '''
    Class for Gaussian-based Control Lyapunov Barrier Functions.
    '''
    def __init__(self, init_value = 0.0, goal = Gaussian(), obstacles = []):
        super().__init__(init_value) 
        self.set_goal(goal)
        self.dim = self.goal_gaussian._dim
        self.obstacle_gaussians = []
        for obs in obstacles:
            if obs._dim != self.dim:
                raise Exception("Dimension of goal Gaussian and obstacle Gaussian must be the same.")
            self.add_obstacle(obs)

    def set_goal(self, goal):
        self.goal_gaussian = goal

    def add_obstacle(self, obstacle):
        self.obstacle_gaussians.append(obstacle)

    def function(self):
        '''
        Gaussian CLBF function.
        '''
        self.goal_gaussian.set_value(self._var)
        self.goal_gaussian.function()

        sum_obs_gaussians = 0.0
        self.goal_gaussian.compute()
        for obs in self.obstacle_gaussians:
            obs.set_value(self._var)
            obs.function()
            sum_obs_gaussians += obs.get_fvalue()
        self._function = - self.goal_gaussian.get_fvalue() + sum_obs_gaussians

    def gradient(self):
        '''
        Gradient of Gaussian CLBF function.
        '''
        self.goal_gaussian.set_value(self._var)
        self.goal_gaussian.gradient()

        sum_obs_gradients =  np.zeros(self.dim)
        for obs in self.obstacle_gaussians:
            obs.set_value(self._var)
            obs.gradient()
            sum_obs_gradients += obs.get_gradient()
        self._gradient = - self.goal_gaussian.get_gradient() + sum_obs_gradients

    def hessian(self):
        '''
        Hessian of Gaussian CLBF function.
        '''
        self.goal_gaussian.set_value(self._var)
        self.goal_gaussian.hessian()

        sum_obs_hessians =  np.zeros([self.dim,self.dim])
        for obs in self.obstacle_gaussians:
            obs.set_value(self._var)
            obs.hessian()
            sum_obs_hessians += obs.get_hessian()
        self._hessian = - self.goal_gaussian.get_hessian() + sum_obs_hessians