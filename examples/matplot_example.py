
import rospy
import math
import numpy as np
import matplotlib.pyplot as plt

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
x_init, y_init = 2.1, 5
initial_state = np.array([x_init,y_init])

# Create CLF
x0 = np.array([0,0])
CLFangle = math.radians(0.0)
lambdav_x, lambdav_y = 4.0, 1.0
CLFeigen = np.array([ lambdav_x , lambdav_y ])
Hv = QuadraticFunction.canonical2D(CLFeigen, CLFangle)
clf = QuadraticLyapunov(state_string, Hv, x0)

# Create CBF
# xaxis_length, yaxis_length = 10.0, 1.0
# lambdah_x, lambdah_y = 1/xaxis_length**2, 1/yaxis_length**2
p0 = np.array([0,3])
CBFangle = math.radians(0.0)
lambdah_x, lambdah_y = 1.0, 1.0
CBFeigen = np.array([ lambdah_x , lambdah_y ])
Hh = QuadraticFunction.canonical2D(CBFeigen, CBFangle)
cbf = QuadraticBarrier(state_string, Hh, p0)

# Create QP controller
lambdav_x_init, lambdav_y_init = -1.0, 4.0
CLFangle_init = math.radians(30.0)
CLFeigen_init = np.array([ lambdav_x_init , lambdav_y_init ])
Hv_init = QuadraticFunction.canonical2D(CLFeigen_init, CLFangle_init)
init_piv = QuadraticFunction.sym2vector(Hv_init)
qp_controller = QPController(plant, clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 1.0], p = [10.0, 10.0], init_pi = init_piv)

# Show initial plot of f(\lambda)
print("Pencil eigenvalues:" + str(qp_controller.pencil_char_roots))
print("Critical:" + str(qp_controller.critical_points))
print("Critical values:" + str(qp_controller.critical_values))

# fig = plt.figure()
# axes_lim = (0, 20.0, 0, 300.0)
# ax = plt.axes(xlim=axes_lim[0:2], ylim=axes_lim[2:4])
# ax.set_title('CLF-CBF QP-based Control')
# lamb = np.arange(axes_lim[0], axes_lim[1], 0.01)
# fvalues = qp_controller.fvalues(lamb)
# ax.plot(lamb, fvalues, zorder=100, color='red')
# plt.show()

# Initialize simulation object
dynamicSimulation = SimulateDynamics(plant, initial_state)

# Simulation loop -------------------------------------------------------------------
print('Running simulation...')
for step in range(0, num_steps):

    # Get simulation state
    state = dynamicSimulation.state()

    # Control
    piv_control, delta_pi = qp_controller.compute_pi_control()
    qp_controller.update_clf_dynamics(piv_control)
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