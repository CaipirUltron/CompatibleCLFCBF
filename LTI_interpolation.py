import sys
import importlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from dynamic_systems import LinearSystem
from functions import QuadraticLyapunov, QuadraticBarrier
from controllers.compatibility import QFunction
from common import hessian_quadratic, rot2D, randomR, genStableLTI, interpolation, is_controllable

''' ----------------------------- Define system ------------------------------ '''
n, m = 2, 1

load_dynamics = False
# load_dynamics = True

if len(sys.argv) > 1 and load_dynamics:

    sim_config = sys.argv[1].replace(".json","")
    sim = importlib.import_module("examples."+sim_config, package=None)
    A, B = sim.plant.A, sim.plant.B 

    def Afun(t): return A
    def Bfun(t): return B

else:

    # A, B = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(1, 5), place=True)

    # A = np.array([[-2, 0],
    #               [ 0,-2]])
    # B = np.array([[1,0],
    #               [0,1]])
    
    A = np.array([[ 0, 1],
                  [-1,-1]])
    B = np.array([[0],
                  [1]])

    def Afun(t): 
        return A
    
    def Bfun(t): 
        return B
    
n, m = A.shape[0], B.shape[1]
plant = LinearSystem(A=Afun(0), B=Bfun(0))

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFcenter = np.zeros(n)

def Hvfun(t):
    eigs1 = np.array([1.0, 8.0])
    eigs2 = np.array([1.0, 8.0])
    eigs_fun = interpolation(eigs1, eigs2)

    angle1, angle2 = 0.0, 180.0
    R1, R2 = rot2D(angle1), rot2D(angle2)
    Rfun = interpolation(R1, R2)
    
    return hessian_quadratic( eigs_fun(t), Rfun(t) )

clf = QuadraticLyapunov(hessian=Hvfun(0), center=CLFcenter)

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFcenter = np.array([6.0, 0.0])

def Hhfun(t):
    eigs1 = np.array([2.0, 1.0])
    eigs2 = np.array([2.0, 1.0])
    eigs_fun = interpolation(eigs1, eigs2)

    angle1, angle2 = 10.0, 10.0
    R1, R2 = rot2D(angle1), rot2D(angle2)
    Rfun = interpolation(R1, R2)
    
    return hessian_quadratic( eigs_fun(t), Rfun(t) )

cbf = QuadraticBarrier(hessian=Hhfun(0), center=CBFcenter)

p = 1.0

''' -------------------------------- Q-function ----------------------------------- '''
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

size = 6.0
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.4*size, size), layout="constrained")
fig.suptitle('Q-function')
ax.set_aspect('auto')

''' ------------------------------- Interpolation ----------------------------------- '''
def update_plot(t):

    print(f"t = {t}")

    if not is_controllable(Afun(t), Bfun(t)):
        raise Exception("System is not controllable.")

    # Hv = Hvfun(t)
    # dotVmatrix = 0.5*(Hv @ A + A.T @ Hv) + p * Hv 
    # eigsdotVmatrix = np.linalg.eigvals(dotVmatrix)
    # eigsdotVmatrix.sort()
    # print(f"CLF condition = {eigsdotVmatrix}")

    qfun.update(M = Mfun(t), N = Nfun(t), H = Hfun(t), w = wfun(t))
    return qfun.plot(ax)

''' ------------------------------- Visualization ----------------------------------- '''
mode = 'animation'
# mode = 'step'
# mode = 'instant'

if len(sys.argv) > 1 and not load_dynamics:
     mode = sys.argv[1]

# Initialize plot 
qfun.init_graphics(ax)
start, stop, step = 0.92, 1.0, 0.001

if mode == 'animation':
    animation = anim.FuncAnimation(fig, func=update_plot, frames=np.arange(start,stop,step), interval=40, repeat=False, blit=False, cache_frame_data=False)

elif mode == 'step':
    t = start
    while t < stop:
        t += step
        update_plot(t)
        plt.pause(0.001)
        plt.waitforbuttonpress()

elif mode == 'instant':
    t = 0.017
    update_plot(t)

plt.show()