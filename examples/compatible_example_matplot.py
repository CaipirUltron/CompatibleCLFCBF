import math
import numpy as np
import matplotlib.pyplot as plt

from system_initialization import plant, clf, cbf, ref_clf
from compatible_clf_cbf.controller import NewQPController
from compatible_clf_cbf.graphical_simulation import SimulationMatplot

# Create QP controller and graphical simulation.
qp_controller = NewQPController(plant, clf, ref_clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 10.0], p = [1.0, 1.0])

try:
    # Simulation loop -------------------------------------------------------------------
    dt = .002
    T = 10
    num_steps = int(T/dt)
    print('Running simulation...')
    for step in range(0, num_steps):

        # Control
        u_control, upi_control = qp_controller.get_control()
        qp_controller.update_clf_dynamics(upi_control)

        # Send actuation commands
        plant.set_control(u_control) 
        plant.actuate(dt)

        # Collect simulation logs ----------------------------------------------------------
        logs = {
            "stateLog": plant.state_log,
            "clfLog": qp_controller.clf_dynamics.state_log
        }
except:
    pass

# Show animation -------------------------------------------------------------------
print('Animating simulation...')
axes_lim = (-6,6,-6,6)
plotSim = SimulationMatplot(axes_lim, 40, logs, clf, cbf, draw_level=True)
plotSim.animate()
plt.show()