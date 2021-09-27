import math
import numpy as np
import matplotlib.pyplot as plt

from system_initialization import plant, clf, ref_clf, cbf
from compatible_clf_cbf.controller import NewQPController

# Create QP controller
qp_controller = NewQPController(plant, clf, ref_clf, cbf)

print("Polar eigenvalues:" + str(qp_controller.pencil_dict["polar_eigenvalues"]))
# print("Alpha = " + str(qp_controller.pencil_dict["alpha"]))
# print("Beta = " + str(qp_controller.pencil_dict["beta"]))

num_points = 100000
min_flambda = -50
max_flambda = 50
phi_var = np.linspace(-math.pi, math.pi, num_points)

fig = plt.figure()
axes_lim = (-math.pi/2, math.pi/2, min_flambda, max_flambda)
ax = plt.axes(xlim=axes_lim[0:2], ylim=axes_lim[2:4])
ax.set_title('polar f-function for the CLF-CBF pair')

lambda_var = np.tan(phi_var)
fvalues = qp_controller.f_values(lambda_var)
ax.plot(phi_var, fvalues, zorder=100, color='red')

x1, y1 = [-math.pi/2, -math.pi/2], [min_flambda-100, max_flambda+100]
x2, y2 = [math.pi/2, math.pi/2], [min_flambda-100, max_flambda+100]
x3, y3 = [-math.pi-1, math.pi+1], [1, 1]
x4, y4 = [-math.pi-1, math.pi+1], [-1, -1]

ax.plot(x1, y1, '--', x2, y2, '--', color='black')
ax.plot(x3, y3, '--', color='green')
ax.plot(x4, y4, '--', color='green')

plt.grid()
plt.show()