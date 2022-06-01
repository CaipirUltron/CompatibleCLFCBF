import numpy as np
from compatible_clf_cbf.dynamic_systems import Function, Gaussian


class CLBF(Function):
    '''
    Class for Gaussian-based Control Lyapunov Barrier Functions.
    '''
    def __init__(self, *args, goal = Gaussian(), obstacles = []):
        super().__init__(*args)
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

    def function(self, point):
        '''
        Gaussian CLBF function.
        '''
        self.goal_gaussian.set_value(*point)
        self.goal_gaussian.function()

        sum_obs_gaussians = 0.0
        self.goal_gaussian.compute()
        for obs in self.obstacle_gaussians:
            obs.set_value(np.array(point))
            obs.function()
            sum_obs_gaussians += obs.get_function()
        self._function = - self.goal_gaussian.get_function() + sum_obs_gaussians

    def gradient(self, point):
        '''
        Gradient of Gaussian CLBF function.
        '''
        self.goal_gaussian.set_value(np.array(point))
        self.goal_gaussian.gradient()

        sum_obs_gradients =  np.zeros(self.dim)
        for obs in self.obstacle_gaussians:
            obs.set_value(np.array(point))
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