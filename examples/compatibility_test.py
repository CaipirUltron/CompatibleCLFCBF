from sage.symbolic.constants import Pi
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from compatible_clf_cbf.controller import QPController
from compatible_clf_cbf.dynamic_simulation import SimulateDynamics
from compatible_clf_cbf.graphical_simulation import GraphicalSimulation
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier, QuadraticFunction
    
# Simulation parameters
dt = .002
sim_freq = 1/dt
T = 20

# Define 2D plant and initial state
f = ['0','0']
g1 = ['1','0']
g2 = ['0','1']
state_string = 'x1, x2'
control_string = 'u1, u2'
plant = AffineSystem(state_string, control_string, f, g1, g2)

# Define initial state for plant simulation
x_init, y_init = 0.1, 5
initial_state = np.array([x_init,y_init])

# Create CLF
lambda_x, lambda_y = 2.0, 1.0
Hv = np.array([ [ lambda_x , 0.0 ],
                [ 0.0 , lambda_y ] ])
x0 = np.array([0,0])
clf = QuadraticLyapunov(state_string, Hv, x0)

# Create CBF
xaxis_length = 1.0
yaxis_length = 1.0
obs_angle = Pi/10
lambda1, lambda2 = 1/xaxis_length**2, 1/yaxis_length**2
Sigma_h = np.array([ [ lambda1 , 0.0 ],
                     [ 0.0 , lambda2 ] ])
R = QuadraticFunction.rot2D(obs_angle)
p0 = np.array([0,1])
cbf = QuadraticBarrier(state_string, Hh, p0)

# Create QP controller
ref = np.array([0,0])
qp_controller = QPController(plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0)

n = 2
print("Basis of R"+str(n)+":")
basis = QuadraticFunction.symmetric_basis(n)
for i in range(np.size(basis,0)):
    print(basis[0])

