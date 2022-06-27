import math
import numpy as np
import matplotlib.pyplot as plt

from system_initialization import plant, clf, cbf
from compatible_clf_cbf.controller import NominalQP
from compatible_clf_cbf.graphical_simulation_matplot import SimulationMatplot
from compatible_clf_cbf.dynamic_systems import sym2vector

# Create QP controller and graphical simulation.
dt = .005
qp_controller = NominalQP(plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0)

# Simulation loop -------------------------------------------------------------------
T = 20
num_steps = int(T/dt)
time = np.zeros(num_steps)
print('Running simulation...')
for step in range(0, num_steps):

    # Simulation time
    time[step] = step*dt

    # Control
    u_control = qp_controller.get_control()
    qp_controller.update_clf_dynamics(np.zeros(qp_controller.sym_dim))
    qp_controller.update_cbf_dynamics(np.zeros(qp_controller.sym_dim))

    # Send actuation commands 
    plant.set_control(u_control) 
    plant.actuate(dt)

# Collect simulation logs ----------------------------------------------------------
logs = {
    "time": time,
    "stateLog": plant.state_log,
    "clfLog": qp_controller.clf.dynamics.state_log,
    "cbfLog": qp_controller.cbf.dynamics.state_log,
    "modeLog": np.zeros(len(time))
}

# Show animation -------------------------------------------------------------------
print('Animating simulation...')
axes_lim = (-6,6,-6,6)
plotSim = SimulationMatplot(axes_lim, 50, logs, clf, cbf, draw_level=True)
plotSim.animate()
# plotSim.plot_frame(2)
plt.show()