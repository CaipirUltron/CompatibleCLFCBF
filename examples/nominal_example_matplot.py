import numpy as np
import matplotlib.pyplot as plt

from system_initialization import plant, initial_state, clf_params, ref_clf_params, cbf_params1, cbf_params2, cbf_params3
from graphical_simulation import SimulationMatplot
from functions import QuadraticLyapunov, QuadraticBarrier
from controllers import NominalQP

# Define quadratic Lyapunov and barriers
clf = QuadraticLyapunov(*initial_state, hessian = clf_params["Hv"], critical = clf_params["x0"])
ref_clf = QuadraticLyapunov(*initial_state, hessian = ref_clf_params["Hv"], critical = ref_clf_params["x0"])

cbf1 = QuadraticBarrier(*initial_state, hessian = cbf_params1["Hh"], critical = cbf_params1["p0"])
cbf2 = QuadraticBarrier(*initial_state, hessian = cbf_params2["Hh"], critical = cbf_params2["p0"])
cbf3 = QuadraticBarrier(*initial_state, hessian = cbf_params3["Hh"], critical = cbf_params3["p0"])

cbfs = [cbf1, cbf2, cbf3]

# Define QP controller
dt = .005
controller = NominalQP(plant, clf, cbfs, gamma = 1.0, alpha = 1.0, p = 10.0)

# Simulation loop -------------------------------------------------------------------
T = 20
num_steps = int(T/dt)
time = np.zeros(num_steps)
print('Running simulation...')
for step in range(0, num_steps):

    # Simulation time
    time[step] = step*dt

    # Control
    u_control = controller.get_control()
    controller.update_clf_dynamics(np.zeros(controller.sym_dim))
    # controller.update_cbf_dynamics(np.zeros(controller.sym_dim))

    # Send actuation commands 
    plant.set_control(u_control) 
    plant.actuate(dt)

# Collect simulation logs ----------------------------------------------------------
logs = {
    "time": time,
    "stateLog": plant.state_log,
    "clfLog": controller.clf.dynamics.state_log,
    # "cbfLog": controller.cbf.dynamics.state_log,
    "modeLog": np.zeros(len(time))
}

# Show animation -------------------------------------------------------------------
print('Animating simulation...')
axes_lim = (-6,6,-6,6)
plotSim = SimulationMatplot(axes_lim, 50, logs, clf, cbfs, draw_level=True)
plotSim.animate()
# plotSim.plot_frame(2)
plt.show()