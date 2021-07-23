import numpy as np
from compatible_clf_cbf.controller import QuadraticProgram

def rotation(phi):
    '''
    Returns 2D rotation matrix.
    '''
    R = np.zeros([2,2])
    R[0,0], R[0,1], R[1,0], R[1,1] = np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi)
    return R


class EllipticalObstacle():
    def __init__( self, obs_points = np.array([[1,2],[1,-1],[-1,-1],[-1,2]]) ):
        '''
        Class for finding the fitting an elliptical barrier function to a bounding box obstacle (4 corner points).
        '''
        self.ecc = 1.0
        self.configure(obs_points)

    def configure(self, obs_points):
        '''
        Configure the object with a new set of four corner points.
        '''
        self.points = obs_points
        self.fit_ellipse()

    def fit_ellipse(self):
        '''
        Fits the ellipse to the four corner points.
        '''
        self.distances = np.zeros(3)
        for i in range(1,4):
            self.distances[i-1] = np.linalg.norm(self.points[0,:] - self.points[i,:])
        indexes = np.argsort(self.distances)
        diag_vec = self.points[0,:] - self.points[indexes[2]+1,:]
        height_vec = self.points[0,:] - self.points[indexes[1]+1,:]
        width_vec = self.points[0,:] - self.points[indexes[0]+1,:]

        # Computes center of the rectangle and angle wrt to the largest dimension
        self.center = 0.5*self.points[0,:] + 0.5*self.points[indexes[2]+1,:]
        self.angle = np.arctan2(height_vec[1], height_vec[0])

        # Computes the ellipse corresponding eigenvalues
        v1 = rotation(self.angle) @ ( self.points[0,:] - self.center ).reshape(2,1)
        v2 = rotation(self.angle) @ ( self.points[1,:] - self.center ).reshape(2,1)
        v3 = rotation(self.angle) @ ( self.points[2,:] - self.center ).reshape(2,1)
        v4 = rotation(self.angle) @ ( self.points[3,:] - self.center ).reshape(2,1)
        Delta = np.array([ [ (v1[0,0])**2 , (v1[1,0])**2 ], 
                           [ (v2[0,0])**2 , (v2[1,0])**2 ], 
                           [ (v3[0,0])**2 , (v3[1,0])**2 ], 
                           [ (v4[0,0])**2 , (v4[1,0])**2 ],
                           [      1     , -self.ecc      ] ])
        solution = np.linalg.pinv(Delta) @ np.array([1,1,1,1,0]).reshape(5,1)
        # solution = np.linalg.pinv(Delta) @ np.array([1,1,1,1]).reshape(4,1)
        self.lambda_x, self.lambda_y = solution[0,0], solution[1,0]

    def compute_barrier(self, position):
        '''
        Compute barrier function and gradient.
        '''
        R = rotation(self.angle)
        delta_x = ( position - self.center ).reshape(2,1)
        H = (R.T) @ np.diag([self.lambda_x, self.lambda_y]) @ R
        h = ( (delta_x.T) @ H @ delta_x )[0,0] - 1 
        gradient_h = 2 * (delta_x.T) @ H

        return h, gradient_h[0,:]


class SafetyControl():
    def __init__(self, obstacles = [ EllipticalObstacle() ], alpha = 1.0, epsilon = [1, 0]):
        '''
        Class for the safety-critical controller.
        '''
        self.state_dim = 2
        self.control_dim = 2

        # Plant and controller parameters
        self.alpha = alpha
        self.epsilon = np.array(epsilon)
        self.Delta = np.array([ [1, -self.epsilon[1]],
                                [ 0, self.epsilon[0] ] ])
        self.set_obstacles(obstacles)

        # Initalize safety critical QP controller
        self.QP_dim = self.control_dim
        self.P = np.eye(self.control_dim)
        self.q = np.zeros(self.control_dim)
        self.qp = QuadraticProgram(P=self.P, q=self.q)

    def set_obstacles(self, obstacles):
        '''
        Sets the elliptical obstacles (change constraints).
        '''
        self.num_obstacles = len(obstacles)
        self.obstacles = obstacles

    def set_nominal(self, u_n):
        '''
        Sets the nominal control (change cost function).
        '''
        self.q = 2 * u_n
        self.qp.set_cost(self.P, self.q)

    def compute_control(self, pose):
        '''
        Sets the cbf constraints based on the passed obstacles.
        '''
        position, orientation = pose[0:1], pose[2]
        if self.num_obstacles != 0:
            A = np.zeros([self.num_obstacles, 2])
            b = np.zeros(self.num_obstacles)
            for i in range(self.num_obstacles):
                obs = self.obstacles[i]
                h_i, gradh_i = obs.compute_barrier(position)
                A[i,:] = - (gradh_i.reshape(2,1)).T @ rotation(orientation) @ self.Delta
                b[i] = self.alpha * h_i
            self.qp.set_constraints(A, b)

        self.qp.solve_QP()

        return self.qp.last_solution