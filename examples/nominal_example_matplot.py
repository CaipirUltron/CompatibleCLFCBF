import math
import numpy as np
import matplotlib.pyplot as plt

from system_initialization import plant, clf, cbf
from compatible_clf_cbf.controller import NominalQP
from compatible_clf_cbf.graphical_simulation import SimulationMatplot

# Create QP controller and graphical simulation.
dt = .005
qp_controller = NominalQP(plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0)

# Simulation loop -------------------------------------------------------------------
T = 20
num_steps = int(T/dt)
print('Running simulation...')
for step in range(0, num_steps):

    # Control
    u_control = qp_controller.get_control()
    qp_controller.update_clf_dynamics(np.zeros(3))

    # Send actuation commands 
    plant.set_control(u_control) 
    plant.actuate(dt)

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