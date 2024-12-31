import sys
import importlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from dynamic_systems import LinearSystem
from functions import QuadraticLyapunov, QuadraticBarrier
from controllers.compatibility import QFunction
from common import is_controllable

example_name = sys.argv[1].replace(".json","")
ex = importlib.import_module("examples.interpolation."+example_name, package=None)

if not all( hasattr(ex, attr) for attr in ["Afun", "Bfun", "Hvfun", "V0_fun", "Hhfun", "h0_fun", "p"] ):
    raise Exception("Loaded example does not define all needed objects.")

''' -------------------- Plant dynamics, CLF, CBF and Q-function ---------------------------- '''

plant = LinearSystem(A=ex.Afun(0), B=ex.Bfun(0))
clf = QuadraticLyapunov(hessian=ex.Hvfun(0), center=ex.V0_fun(0))
cbf = QuadraticBarrier(hessian=ex.Hhfun(0), center=ex.h0_fun(0))

qfun = QFunction(plant, clf, cbf)

''' -------------------------- Methods for Q-function update -------------------------------- '''
def Mfun(t):
    B = ex.Bfun(t)
    G = B @ B.T
    M = G @ ex.Hhfun(t)
    return M

def Nfun(t):
    B = ex.Bfun(t)
    G = B @ B.T
    N = ex.p * G @ ex.Hvfun(t) - ex.Afun(t)    
    return N

def Hfun(t):
    return ex.Hhfun(t)

def wfun(t):
    N = Nfun(t)
    return N @ ( ex.h0_fun(t) - ex.V0_fun(t) )

''' ----------------------------- Q-function figure --------------------------------- '''
size = 6.0
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.4*size, size), layout="constrained")
fig.suptitle('Q-function')
ax.set_aspect('auto')

''' --------------------------- Interpolation update -------------------------------- '''
def update_plot(t):

    print(f"t = {t}")

    if not is_controllable(ex.Afun(t), ex.Bfun(t)):
        raise Exception("System is not controllable.")

    qfun.update(M = Mfun(t), N = Nfun(t), H = Hfun(t), w = wfun(t))
    return qfun.plot(ax)

''' ------------------------------- Visualization ----------------------------------- '''
mode = 'animate'
# mode = 'step'
# mode = 'instant'

start, stop, step = 0.92, 1.0, 0.001
t = 0.017

if len(sys.argv) > 2:
     mode = sys.argv[2]

qfun.init_graphics(ax)

if mode == 'animate':
    animation = anim.FuncAnimation(fig, func=update_plot, frames=np.arange(start,stop,step), interval=40, repeat=False, blit=False, cache_frame_data=False)

elif mode == 'step':
    t = start
    while t < stop:
        t += step
        update_plot(t)
        plt.pause(0.001)
        plt.waitforbuttonpress()
        if not plt.fignum_exists(fig.number):
            sys.exit()

elif mode == 'instant':
    update_plot(t)

plt.show()