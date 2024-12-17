import json
import control
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from controllers.compatibility import MatrixPencil, QFunction
from common import hessian_quadratic, vector2sym, rot2D, randomR, genStableLTI, interpolation

load = True

''' ------------------------------------- Define system ------------------------------------- '''
n, m = 2, 2

A0, B0 = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(1, 5), place=True)
A1, B1 = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(1, 5), place=True)

if load:
    with open("logs/interesting.json") as file:
        print("Loading interesting example...")
        example = json.load(file)

    A0, B0 = np.array(example["A0"]), np.array(example["B0"])
    A1, B1 = np.array(example["A1"]), np.array(example["B1"])

Afun = interpolation(A0, A1)
Bfun = interpolation(B0, B1)

# Afun = interpolation(np.zeros((n,n)), np.zeros((n,n)))
# Bfun = interpolation(np.eye(n),np.eye(n))

''' ---------------------------- Define quadratic CLF and CBF ----------------------------------- '''

# CLFeigs1 = np.random.randint(low=1, high=10, size=n)
# Hv0 = hessian_quadratic( CLFeigs1, randomR(n) )

# CLFeigs2 = np.random.randint(low=1, high=10, size=n)
# Hv1 = hessian_quadratic( CLFeigs2, randomR(n) )
# CLFcenter = np.zeros(n)

# CBFeigs = np.random.randint(low=1, high=10, size=n)
# CBFcenter = 5*np.random.randn(n)
# Hh = hessian_quadratic(CBFeigs, randomR(n) )

CLFeigs1 = np.array([ 10.0, 1.0 ])
CLFeigs2 = np.array([ 1.0, 10.0 ])

angle0 = np.deg2rad(-180)
angle1 = np.deg2rad(180)

CLFcenter = np.zeros(n)

CBFeigs = np.array([ 1.0, 10.0 ])
CBFcenter = np.array([5.0, 1.0])
Hh = hessian_quadratic(CBFeigs, rot2D(0) )

CLFeigs_fun = interpolation(CLFeigs1, CLFeigs2)
angle_fun = interpolation(angle0, angle1)
Hhfun = interpolation(Hh, Hh)

p = 1.0

''' ---------------------------- Pencil and Q-function ----------------------------------- '''
def Hvfun(t):
    return hessian_quadratic( CLFeigs1, rot2D(angle_fun(t)) )

def Mfun(t):
    B = Bfun(t)
    G = B @ B.T
    return G @ Hh

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
for k, eig in enumerate( pencil.eigens ):
    print(f"{k+1}-th gen. eigenvalue of P(λ) = {eig.eigenvalue}.")

qfun = QFunction(pencil, Hfun(0), wfun(0))
for k, eig in enumerate( qfun.Smatrix_pencil.eigens ):
    print(f"{k+1}-th gen. eigenvalue of S(λ) = {eig.eigenvalue}")

''' ------------------------------- Interpolation ----------------------------------- '''
def update_plot(t):

    A = Afun(t)
    B = Bfun(t)

    rankC = np.linalg.matrix_rank( control.ctrb(A,B) )
    if rankC < n:
        raise Exception("Systems are not controllable.")

    qfun.update(M = Mfun(t), N = Nfun(t), H = Hfun(t), w = wfun(t))
    return qfun.plot()

''' --------------------------------- Animation ------------------------------------- '''

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
fig.suptitle('Q-function plot')

# Plot before compatibilization
qfun.init_graphics(ax)

start, stop, step = 0.35, 0.6, 0.01
start, stop, step = 0.87, 0.95, .0001
fps = 60
animation = anim.FuncAnimation(fig, func=update_plot, frames=np.arange(start,stop,step), interval=10, repeat=False, blit=True, cache_frame_data=False)

plt.show()

print("Save results? Y/N:")

example = {"A0": A0.tolist(), "A1": A1.tolist(), "B0": B0.tolist(), "B1": B1.tolist()}
with open("logs/interesting.json", "w") as file:
    print("Saving interesting example...")
    json.dump(example, file, indent=4)

# ''' --------------------------------- Test compatibilization ------------------------------------- '''
# def Hvfun(var):
#     eps = 0.01
#     L = vector2sym(var)
#     return L @ L.T + eps * np.eye(n)

# clf_dict = {"Hv_fun": Hvfun, "center": CLFcenter, "Hv": Hvfun(0) }

# results = qfun.compatibilize( plant, clf_dict, p=p )