import math
import numpy as np
import matplotlib.pyplot as plt

from system_initialization import plant, clf, cbf, ref_clf
from compatible_clf_cbf.controller import NewQPController
from compatible_clf_cbf.graphical_simulation import SimulationMatplot

# Create QP controller and graphical simulation.
dt = .002
qp_controller = NewQPController(plant, clf, ref_clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 10.0], p = [1.0, 1.0], dt = dt)

# Simulation loop -------------------------------------------------------------------
T = 20
num_steps = int(T/dt)
print('Running simulation...')
for step in range(0, num_steps):

    # Control
    u_control, upi_control = qp_controller.get_control()
    # upi_control = np.zeros(3)

    # Send actuation commands
    qp_controller.update_clf_dynamics(upi_control)
    plant.set_control(u_control) 
    plant.actuate(dt)

    # print("CLF = " + str(qp_controller.V))
    # print("CBF = " + str(qp_controller.h))
    # print("Rate CLF = " + str(qp_controller.Vpi))

    # print("Compatibility Barrier 1 = " + str(qp_controller.h_gamma1))
    # print("Compatibility Barrier 2 = " + str(qp_controller.h_gamma2))
    # print("Compatibility Barrier 3 = " + str(qp_controller.h_gamma3))

    # Collect simulation logs ----------------------------------------------------------
    logs = {
        "stateLog": plant.state_log,
        "clfLog": qp_controller.clf_dynamics.state_log
    }

# Show animation -------------------------------------------------------------------
print('Animating simulation...')
axes_lim = (-6,6,-6,6)
plotSim = SimulationMatplot(axes_lim, 80, logs, clf, cbf, draw_level=True)
plotSim.animate()
plt.show()