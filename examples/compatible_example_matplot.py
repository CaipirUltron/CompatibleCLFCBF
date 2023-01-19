import numpy as np
import matplotlib.pyplot as plt

from examples.system_initialization import plant, initial_state, clf_params, ref_clf_params, cbf_params1, cbf_params2, cbf_params3
from graphical_simulation import SimulationMatplot
from functions import QuadraticLyapunov, QuadraticBarrier
from controllers import CompatibleQPController

# Define quadratic Lyapunov and barriers
clf = QuadraticLyapunov(*initial_state, hessian = clf_params["Hv"], critical = clf_params["x0"])
ref_clf = QuadraticLyapunov(*initial_state, hessian = ref_clf_params["Hv"], critical = ref_clf_params["x0"])

cbf1 = QuadraticBarrier(*initial_state, hessian = cbf_params1["Hh"], critical = cbf_params1["p0"])
cbf2 = QuadraticBarrier(*initial_state, hessian = cbf_params2["Hh"], critical = cbf_params2["p0"])
cbf3 = QuadraticBarrier(*initial_state, hessian = cbf_params3["Hh"], critical = cbf_params3["p0"])

cbfs = [cbf1, cbf2, cbf3]

# Create QP controller and graphical simulation.
dt = .005
controller = CompatibleQPController(plant, clf, ref_clf, cbfs, gamma = [1.0, 10.0], alpha = [1.0, 10.0], p = [1.0, 1.0], dt = dt)

# Simulation loop -------------------------------------------------------------------
T = 10
num_steps = int(T/dt)
time = np.zeros(num_steps+1)
print('Running simulation...')
for step in range(1, num_steps+1):

    # Simulation time
    time[step] = step*dt    

    # Inner loop control
    u_control = controller.get_control()

    # Outer loop control
    upi_control = controller.get_clf_control()

    # Send actuation commands
    controller.update_clf_dynamics(upi_control)
    print("Active CBF = " + str(controller.active_cbf_index()))

    plant.set_control(u_control)
    plant.actuate(dt)

# Collect simulation logs ----------------------------------------------------------
logs = {
    "time": time,
    "state": plant.state_log,
    "control": plant.control_log,
    "clf": controller.clf.dynamics.state_log,
    "mode": controller.mode_log,
    "equilibria": controller.equilibrium_points
}

# Show animation -------------------------------------------------------------------
print('Printing simulation...')
axes_lim = (-6,6,-6,6)

plotSim = SimulationMatplot(axes_lim, 80, logs, clf, cbfs, draw_level=True)
plotSim.animate()
# plotSim.plot_frame(0.1)
plt.show()