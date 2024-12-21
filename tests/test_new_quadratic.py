import numpy as np
import matplotlib.pyplot as plt
from controllers.compatibility import MatrixPencil, QFunction
from controllers import CompatibleQP

from dynamic_systems import LinearSystem, DriftLess
from common import hessian_quadratic, rot2D, vector2sym
from functions import QuadraticLyapunov, QuadraticBarrier

limits = 12*np.array((-1,1,-1,1))

''' ----------------------------------- Define LTI system ---------------------------------- '''
n = 2
x0 = np.array([2,6])
A = np.zeros((n,n))
B = np.eye(n)

plant = LinearSystem(x0, np.zeros(2), A=A, B=B)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFaxes = np.array([1.0, 8.0])
CLFangle = 0.0
CLFcenter = np.zeros(2)
clf = QuadraticLyapunov.geometry2D(CLFaxes, CLFangle, CLFcenter, level=1, limits=limits)
Hv = clf.H

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFaxes = [1.0, 2.0]
CBFangle = 30.0
CBFcenter = np.array([0.0, 5.0])
cbf = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter, limits=limits)
Hh = cbf.H

cbfs = [cbf]
num_cbfs = len(cbfs)
''' ----------------------------- Compatibilization --------------------------------- '''

controller = CompatibleQP(plant, clf, cbfs)

Hvs = controller.compatibilize()
clf.set_params(hessian=Hvs[0])
clf.generate_contour()

''' --------------------------------- Plot ------------------------------------- '''
fig, axes = plt.subplots(ncols=1+num_cbfs, nrows=1, figsize=(6.0*(1+num_cbfs), 5.0))
fig.suptitle('Compatible Quadratic CLF-CBFs')

clf.plot_levels(axes[0], levels=[20.0], color='b')
cbf.plot_levels(axes[0], levels=[0.0], color='g')

axes[0].set_xlim(limits[0:2])
axes[0].set_ylim(limits[2:])
axes[0].legend()

for k, qfun in enumerate(controller.Qfunctions):
    qfun.init_graphics(axes[k+1])
    qfun.plot()

plt.show()