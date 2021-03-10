
import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from compatible_clf_cbf.controller import QPController
from compatible_clf_cbf.dynamic_simulation import SimulateDynamics
from compatible_clf_cbf.graphical_simulation import SimulationMatplot
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier, QuadraticFunction

# Simulation parameters
dt = .005
T = 10
num_steps = int(T/dt)

# Define 2D plant and initial state
f = ['0','0']
g1 = ['1','0']
g2 = ['0','1']
g = [g1,g2]
state_string = 'x1, x2, '
control_string = 'u1, u2, '
plant = AffineSystem(state_string, control_string, f, *g)

# Define initial state for plant simulation
x_init, y_init = 3.1, 5
initial_state = np.array([x_init,y_init])

# Create CLF
lambdav_x, lambdav_y = 1.0, 2.0
CLFangle = math.radians(45.0)
x0 = np.array([0,0])

CLFeigen = np.array([ lambdav_x , lambdav_y ])
Hv = QuadraticFunction.canonical2D(CLFeigen, CLFangle)
clf = QuadraticLyapunov(state_string, Hv, x0)

# Create CBF
xaxis_length, yaxis_length = 2.0, 1.0
CBFangle = math.radians(-45.0)
p0 = np.array([3,3])

lambdah_x, lambdah_y = 1/xaxis_length**2, 1/yaxis_length**2
CBFeigen = np.array([ lambdah_x , lambdah_y ])
Hh = QuadraticFunction.canonical2D(CBFeigen, CBFangle)
cbf = QuadraticBarrier(state_string, Hh, p0)

# Create QP controller
init_eig = CLFeigen
qp_controller = QPController(plant, clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 1.0], p = [10.0, 10.0], init_eig = init_eig)

# Initialize simulation object
dynamicSimulation = SimulateDynamics(plant, initial_state)

# Simulation loop -------------------------------------------------------------------
print('Running simulation...')
for step in range(0, num_steps):

    # Get simulation state
    state = dynamicSimulation.state()

    # Control
    lambda_control, delta_pi = qp_controller.compute_lambda_control()
    qp_controller.update_clf_dynamics(lambda_control)
    control, delta = qp_controller.compute_control(state)

    # Send actuation commands 
    dynamicSimulation.send_control_inputs(control, dt)

# Collect simulation logs ----------------------------------------------------------
logs = {
    "stateLog": dynamicSimulation.state_log,
    "clfLog": qp_controller.clf_dynamics.state_log
}

# Show animation -------------------------------------------------------------------
print('Animating simulation...')
axes_lim = (-6,6,-6,6)
plotSim = SimulationMatplot(axes_lim, 40, logs, clf, cbf, draw_level=True)
plotSim.animate()
plt.show()