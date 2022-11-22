import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# from examples.nominal_example_matplot import logs
from examples.compatible_example_matplot import logs

time = logs["time"]

state_x = logs["state"][0]
state_y = logs["state"][1]
all_states = np.hstack([state_x, state_y])

control_x = logs["control"][0]
control_y = logs["control"][1]
all_controls = np.hstack([control_x, control_y])

max_time = 10
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
# fig.suptitle('Path variables')
# ax1.set_aspect('equal', adjustable='box')
ax1.set_title('State')
ax1.plot(time, state_x, "--", label='$x_1$', linewidth=2, markersize=10)
ax1.plot(time, state_y, "--", label='$x_2$', linewidth=2, markersize=10)
ax1.legend(fontsize=12)
ax1.set_xlim(0, max_time)
ax1.set_ylim(np.min(all_states)-1, np.max(all_states)+1)
ax1.set_xlabel('Time [s]')
# plt.grid()

ax2 = fig.add_subplot(122)
ax2.set_title('Control')
ax2.plot(time, control_x, "--", label='$u_1$', linewidth=2, markersize=10, alpha=1.0)
ax2.plot(time, control_y, "--", label='$u_2$', linewidth=2, markersize=10, alpha=0.6) 
ax2.legend(fontsize=12)
ax2.set_xlim(0, max_time)
ax2.set_ylim(np.min(all_controls)-1, np.max(all_controls)+1)
ax2.set_xlabel('Time [s]')
# plt.grid()

plt.savefig('plot_simulation1.eps', format='eps', transparent=True)
# plt.savefig("plot_nominal_simulation.svg", format="svg",transparent=True)

plt.show()