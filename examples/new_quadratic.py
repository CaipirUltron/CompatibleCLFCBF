import numpy as np
import matplotlib.pyplot as plt
from controllers import CompatibleQP

from dynamic_systems import LinearSystem, DriftLess
from functions import QuadraticLyapunov, QuadraticBarrier

limits = 12*np.array((-1,1,-1,1))

''' --------------------------------- Define LTI system ------------------------------------- '''
n = 2
x0 = np.array([-4,9])

A = np.zeros((n,n))
B = np.eye(n)
plant = LinearSystem(x0, np.zeros(2), A=A, B=B)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFaxes = np.array([1.0, 4.0])
CLFangle = 0.0
CLFcenter = np.zeros(2)
clf = QuadraticLyapunov.geometry2D(CLFaxes, CLFangle, CLFcenter, level=1, limits=limits)
Hv = clf.H

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFaxes = [2.0, 1.0]
CBFangle = 10.0
CBFcenter = np.array([0.0, 5.0])
cbf = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter, limits=limits)
Hh = cbf.H

cbfs = [cbf]
num_cbfs = len(cbfs)

''' --------------------------- Compatible controller --------------------------------- '''
T = 15
sample_time = 5e-3
controller = CompatibleQP(plant, clf, cbfs, alpha = [3.0, 10.0], beta = [10.0, 10.0], p = [1.0, 1.0], dt = sample_time)

Hvs = controller.compatibilize(verbose=True)
Hv = Hvs[0]

''' ------------------------------ Configure plot ----------------------------------- '''
plot_config = {
    "figsize": (5,5), 
    "gridspec": (1,1,1), 
    "widthratios": [1], 
    "heightratios": [1], 
    "limits": limits,
    "path_length": 10, 
    "numpoints": 1000, 
    "drawlevel": True, 
    "resolution": 50, 
    "fps":30, "pad":2.0, 
    "invariant": True, 
    "equilibria": True, 
    "arrows": False,
}

logs = { "sample_time": sample_time }

# for k, qfun in enumerate(controller.Qfunctions):
#     h = qfun.composite_barrier()
#     print(f"Found Hv = \n {Hv}")
#     print(f"{k+1} Q-function h = {h}")

# clf.set_params(hessian=Hv)
# clf.generate_contour()

# ''' --------------------------------- Plot ------------------------------------- '''
# fig, axes = plt.subplots(ncols=1+num_cbfs, nrows=1, figsize=(6.0*(1+num_cbfs), 5.0))
# fig.suptitle('Compatible Quadratic CLF-CBFs')

# clf.plot_levels(axes[0], levels=[20.0], color='b')

# for k, cbf in enumerate(cbfs):
#     cbf.plot_levels(axes[0], levels=[0.0], color='g')
#     eqs = controller.get_cbf_equilibria(k)
#     for eq in eqs:
#         h = cbf(eq["point"])
#         print(f"h at equilibrium point = {h}")

# x_eq_array = [ eq["point"][0] for eq in controller.equilibrium_points ]
# y_eq_array = [ eq["point"][1] for eq in controller.equilibrium_points ]
# axes[0].plot(x_eq_array, y_eq_array, "o")

# axes[0].set_xlim(limits[0:2])
# axes[0].set_ylim(limits[2:4])

# for k, qfun in enumerate(controller.Qfunctions):
#     qfun.init_graphics(axes[k+1])
#     qfun.plot()

# plt.show()