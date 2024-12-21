import json, sys

import control
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from controllers.compatibility import MatrixPencil, QFunction
from common import hessian_quadratic, vector2sym, rot2D, randomR, genStableLTI, interpolation

''' ----------------------------- Define system (varying A, B) ------------------------------ '''
n, m = 2, 2

# A, B = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(1, 5), place=True)

A = np.zeros((n,n))
B = np.eye(n)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFeigs = np.array([10.0, 1.0])

angle1 = np.deg2rad(0.0)
angle2 = np.deg2rad(90.0)
angle_fun = interpolation(angle1, angle2)

CLFcenter = np.zeros(n)

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFeigs = np.array([1.0, 10.0])
Rh = rot2D(np.deg2rad(0.0))
CBFcenter = np.array([0.0, 5.0])

p = 1.0

''' ---------------------------- Pencil and Q-function ----------------------------------- '''
Afun = lambda t: A
Bfun = lambda t: B

def Hvfun(t):
    return hessian_quadratic( CLFeigs, rot2D(angle_fun(t)) )

def Hhfun(t):
    return hessian_quadratic( CBFeigs, Rh )

def Mfun(t):
    B = Bfun(t)
    G = B @ B.T
    return G @ Hhfun(t)

def Nfun(t):
    B = Bfun(t)
    G = B @ B.T
    return p * G @ Hvfun(t) - Afun(t)

def Hfun(t):
    return Hhfun(t)

def wfun(t):
    N = Nfun(t)
    return N @ ( CBFcenter - CLFcenter )

pencil = MatrixPencil(Mfun(0), Nfun(0))
qfun = QFunction(pencil, Hfun(0), wfun(0))

''' ------------------------------- Interpolation ----------------------------------- '''
def update_plot(t):

    # print(f"t = {t}")

    rankC = np.linalg.matrix_rank( control.ctrb(Afun(t), Bfun(t)) )
    if rankC < n:
        raise Exception("Systems are not controllable.")

    qfun.update(M = Mfun(t), N = Nfun(t), H = Hfun(t), w = wfun(t))

    print("")
    for k, root in enumerate(qfun.zero_poly.roots()):
        h = qfun.composite_barrier(root)
        print(f"h(λ{k+1}) = {h}")

    return qfun.plot()

''' --------------------------------- Animation ------[------------------------------- '''

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
fig.suptitle('Q-function plot')

# Initialize plot 
qfun.init_graphics(ax)

start, stop, step = 0.0, 1.0, 0.001
animation = anim.FuncAnimation(fig, func=update_plot, frames=np.arange(start,stop,step), interval=300, repeat=False, blit=True, cache_frame_data=False)

update_plot(0.1)

for k, eig in enumerate( pencil.eigens ):
    print(f"{k+1}-th gen. eigenvalue of P(λ) = {eig.eigenvalue}")

print("")
for k, eig in enumerate( qfun.Smatrix_pencil.eigens ):
    print(f"{k+1}-th gen. eigenvalue of S(λ) = {eig.eigenvalue}")

print("")
for k, root in enumerate( qfun.dSdet.roots() ):
    print(f"{k+1}-th root of |S(λ)|' = {root}")

qfun.plot()

plt.show()