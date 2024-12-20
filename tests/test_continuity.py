import json, sys

import control
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from controllers.compatibility import MatrixPencil, QFunction
from common import hessian_quadratic, vector2sym, rot2D, randomR, genStableLTI, interpolation

load = True

''' ----------------------------- Define system (varying A, B) ------------------------------ '''
n, m = 4, 4

A0, B0 = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(1, 5), place=True)
A1, B1 = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(1, 5), place=True)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFeigs1 = np.random.randint(low=1, high=10, size=n)
CLFeigs2 = np.random.randint(low=1, high=10, size=n)
Rv = randomR(n)
CLFcenter = np.zeros(n)

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFeigs1 = np.random.randint(low=1, high=10, size=n)
CBFeigs2 = np.random.randint(low=1, high=10, size=n)
Rh = randomR(n)
CBFcenter = 5*np.random.randn(n)

p = 1.0

if load:
    
    with open("logs/interesting.json") as file:
        print("Loading interesting example...")
        example = json.load(file)

    # Load system parameters
    n, m = example["n"], example["m"]
    A0, B0 = np.array(example["A0"]), np.array(example["B0"])
    A1, B1 = np.array(example["A1"]), np.array(example["B1"])

    # Load CLF parameters
    CLFeigs1 = np.array(example["CLFeigs1"])
    CLFeigs2 = np.array(example["CLFeigs2"])
    Rv = np.array(example["Rv"])
    CLFcenter = np.array(example["CLFcenter"])

    # Load CBF parameters
    CBFeigs1 = np.array(example["CBFeigs1"])
    CBFeigs2 = np.array(example["CBFeigs2"])
    Rh = np.array(example["Rh"])
    CBFcenter = np.array(example["CBFcenter"])

    p = example["p"]

Afun = interpolation(A0, A1)
Bfun = interpolation(B0, B1)

CLFeigs_fun = interpolation(CLFeigs1, CLFeigs2)
CBFeigs_fun = interpolation(CBFeigs1, CBFeigs2)

''' ---------------------------- Pencil and Q-function ----------------------------------- '''

def Hvfun(t):
    return hessian_quadratic( CLFeigs_fun(t), Rv )

def Hhfun(t):
    return hessian_quadratic( CBFeigs_fun(t), Rh )

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

    h = qfun.composite_barrier()
    print(f"h = {h}")

    return qfun.plot()

''' --------------------------------- Animation ------[------------------------------- '''

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
fig.suptitle('Q-function plot')

# Initialize plot 
qfun.init_graphics(ax)

# start, stop, step = 0.10, 0.4, 0.001
# start, stop, step = 0.87, 0.95, 0.0001
# start, stop, step = 0.22, 0.25, 0.0001
start, stop, step = 0.72, 1.0, 0.001

fps = 60
animation = anim.FuncAnimation(fig, func=update_plot, frames=np.arange(start,stop,step), interval=300, repeat=False, blit=True, cache_frame_data=False)

# update_plot(0.8953)
# update_plot(0.725)

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

''' --------------------------- Save interesting results ---------------------------- '''

pressed = input("Save results? Y/N:")
if pressed.lower() == 'y':
    
    example = {"n": n, "m": m, "A0": A0.tolist(), "A1": A1.tolist(), "B0": B0.tolist(), "B1": B1.tolist(),
               "CLFeigs1": CLFeigs1.tolist(), "CLFeigs2": CLFeigs2.tolist(), "Rv": Rv.tolist(), "CLFcenter": CLFcenter.tolist(), 
               "CBFeigs1": CBFeigs1.tolist(), "CBFeigs2": CBFeigs2.tolist(), "Rh": Rh.tolist(), "CBFcenter": CBFcenter.tolist(),
               "p": p }
    
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