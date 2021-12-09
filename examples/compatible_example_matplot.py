import math
import numpy as np
import matplotlib.pyplot as plt

from system_initialization import plant, clf, cbf, ref_clf
from compatible_clf_cbf.controller import NewQPController
from compatible_clf_cbf.graphical_simulation import SimulationMatplot

# Create QP controller and graphical simulation.
dt = .005
qp_controller = NewQPController(plant, clf, ref_clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 10.0], p = [1.0, 1.0], dt = dt)

# Simulation loop -------------------------------------------------------------------
T = 20
num_steps = int(T/dt)
time = np.zeros(num_steps)
print('Running simulation...')
for step in range(0, num_steps):

    # Simulation time
    time[step] = step*dt

    # Inner loop control
    u_control = qp_controller.get_control()

    # Outer loop control
    upi_control = qp_controller.get_clf_control()

    # print("f-function zeros:" + str(np.arctan(qp_controller.f_dict["zeros"])))

    # Send actuation commands
    qp_controller.update_clf_dynamics(upi_control)
    qp_controller.update_cbf_dynamics(np.zeros(3))

    plant.set_control(u_control)
    plant.actuate(dt)

    # Collect simulation logs ----------------------------------------------------------
    logs = {
        "time": time,
        "stateLog": plant.state_log,
        "clfLog": qp_controller.clf_dynamics.state_log,
        "cbfLog": qp_controller.cbf_dynamics.state_log
    }

# Show animation -------------------------------------------------------------------
print('Animating simulation...')
axes_lim = (-6,6,-6,6)
plotSim = SimulationMatplot(axes_lim, 80, logs, clf, cbf, draw_level=True)
plotSim.animate()
plt.show()