import numpy as np
import matplotlib.pyplot as plt

from dynamic_systems import LinearSystem
from functions import QuadraticLyapunov, QuadraticBarrier
from controllers.compatibility import MatrixPencil, QFunction
from common import hessian_quadratic, vector2sym, rot2D, genStableLTI

n, m = 2, 2
# A, B = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(1, 5), place=True)

x0 = np.array([2,6])
A = np.zeros((n,n))
B = np.eye(n)

plant = LinearSystem(x0, np.zeros(2), A=A, B=B)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
# CLFeigs = np.array([10.0, 1.0])
# Rv = rot2D(np.deg2rad(1.0))
# Hv = hessian_quadratic( CLFeigs, Rv )
# CLFcenter = np.zeros(n)

CLFaxes = np.array([1.0, 4.0])
CLFangle = 0.0
CLFcenter = np.zeros(2)
clf = QuadraticLyapunov.geometry2D(CLFaxes, CLFangle, CLFcenter, level=1)
Hv = clf.H

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
# CBFeigs = np.array([1.0, 10.0])
# Rh = rot2D(np.deg2rad(0.0))
# Hh = hessian_quadratic( CBFeigs, Rh )
# CBFcenter = np.array([0.0, 5.0])

CBFaxes = [1.0, 2.0]
CBFangle = 30.0
CBFcenter = np.array([0.0, 5.0])
cbf = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter)
Hh = cbf.H

p = 1.0

''' --------------------------------- Q-function ------------------------------------ '''
qfun = QFunction(plant, clf, cbf, p)

''' --------------------------------- Animation ------------------------------------- '''

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
fig.suptitle('Q-function plot')

''' ----------------------------- Compatibilization --------------------------------- '''
h = qfun.composite_barrier()
print(f"h(Hv) = {h}")

results = qfun.compatibilize(verbose=True)
Hv = results["Hv"]

# eigHv, Rv = np.linalg.eig(Hv)
# print(f"Eigs Hv = {eigHv}")
# print(f"Rot Hv = {Rv}")

qfun.init_graphics(ax)
qfun.plot()

plt.show()