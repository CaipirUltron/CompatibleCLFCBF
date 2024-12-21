import numpy as np
import matplotlib.pyplot as plt

from controllers.compatibility import MatrixPencil, QFunction
from common import hessian_quadratic, vector2sym, rot2D, genStableLTI

n, m = 2, 2
# A, B = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(1, 5), place=True)

A = np.zeros((n,n))
B = np.eye(n)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFeigs = np.array([10.0, 1.0])
Rv = rot2D(np.deg2rad(1.0))
Hv = hessian_quadratic( CLFeigs, Rv )
CLFcenter = np.zeros(n)

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFeigs = np.array([1.0, 10.0])
Rh = rot2D(np.deg2rad(0.0))
Hh = hessian_quadratic( CBFeigs, Rh )
CBFcenter = np.array([0.0, 5.0])

p = 1.0

''' ---------------------------- Pencil and Q-function ----------------------------------- '''
G = B @ B.T
M = G @ Hh
N = p * G @ Hv - A
w = N @ ( CBFcenter - CLFcenter )

pencil = MatrixPencil(M, N)
qfun = QFunction(pencil, Hh, w)

''' --------------------------------- Animation ------------------------------------- '''

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
fig.suptitle('Q-function plot')

''' ----------------------------- Compatibilization --------------------------------- '''
h = qfun.composite_barrier()
print(f"h(Hv) = {h}")

def Hvfun(var):
    eps = 1e-3
    L = vector2sym(var)
    return L @ L.T + eps * np.eye(n)
clf_dict = {"Hv_fun": Hvfun, "center": CLFcenter, "Hv": Hv }

results = qfun.compatibilization( A, B, clf_dict, p=p )
print("Compatibilization results:")
print(results)

Hv = results["Hv"]
eigHv, Rv = np.linalg.eig(Hv)
print(f"Eigs Hv = {eigHv}")
print(f"Rot Hv = {Rv}")

qfun.init_graphics(ax)
qfun.plot()
# qfun.plot_contours(ax)

plt.show()