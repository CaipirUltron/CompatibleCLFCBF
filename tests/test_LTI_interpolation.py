import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from dynamic_systems import LinearSystem
from functions import QuadraticLyapunov, QuadraticBarrier
from controllers.compatibility import QFunction
from common import hessian_2Dquadratic, vector2sym, rot2D, randomR, genStableLTI, interpolation, is_controllable

''' ----------------------------- Define system (varying A, B) ------------------------------ '''
n, m = 2, 1

# A, B = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(1, 5), place=True)

# A = np.zeros((n,n))
# B = np.eye(n)

A = np.array([[ 0, 1],
              [-1,-1]])
B = np.array([[0],
              [1]])

Afun = lambda t: A
Bfun = lambda t: B
plant = LinearSystem(np.zeros(n), np.zeros(m), A=A, B=B)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFeigs = np.array([1.0, 8.0])
angle1, angle2 = np.deg2rad(0.0), np.deg2rad(180.0)
angle_fun = interpolation(angle1, angle2)
CLFcenter = np.zeros(n)

def Hvfun(t):
    Hv = hessian_2Dquadratic( CLFeigs, angle_fun(t) )
    return Hv

clf = QuadraticLyapunov(hessian=Hvfun(0), center=CLFcenter)

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFeigs = np.array([2.0, 1.0])
CBFangle = np.deg2rad(10.0)
CBFcenter = np.array([6.0, 0.0])

def Hhfun(t):
    Hh = hessian_2Dquadratic( CBFeigs, CBFangle )
    return Hh

cbf = QuadraticBarrier(hessian=Hhfun(0), center=CBFcenter)

p = 1.0

''' ---------------------------- Pencil and Q-function ----------------------------------- '''
def Mfun(t):
    B = Bfun(t)
    G = B @ B.T
    M = G @ Hhfun(t)
    return M

def Nfun(t):
    B = Bfun(t)
    G = B @ B.T
    N = p * G @ Hvfun(t) - Afun(t)    
    return N

def Hfun(t):
    return Hhfun(t)

def wfun(t):
    N = Nfun(t)
    return N @ ( CBFcenter - CLFcenter )

qfun = QFunction(plant, clf, cbf)

''' ------------------------------- Interpolation ----------------------------------- '''
def update_plot(t):

    print(f"t = {t}")

    if not is_controllable(Afun(t), Bfun(t)):
        raise Exception("System is not controllable.")

    Hv = Hvfun(t)
    dotVmatrix = 0.5*(Hv @ A + A.T @ Hv) + p * Hv 
    eigsdotVmatrix = np.linalg.eigvals(dotVmatrix)
    eigsdotVmatrix.sort()
    print(f"CLF condition = {eigsdotVmatrix}")

    qfun.update(M = Mfun(t), N = Nfun(t), H = Hfun(t), w = wfun(t))
    return qfun.plot()

''' --------------------------------- Animation ------[------------------------------- '''
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
fig.suptitle('Q-function Interpolation')

# Initialize plot 
qfun.init_graphics(ax)

start, stop, step = 0.0, 1.0, 0.01
animation = anim.FuncAnimation(fig, func=update_plot, frames=np.arange(start,stop,step), interval=80, repeat=False, blit=True, cache_frame_data=False)

t = 0.148
update_plot(t)

Hv = Hvfun(t)
print(Hv)

qfun.plot()
plt.show()