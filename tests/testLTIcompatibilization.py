import numpy as np
import control
import matplotlib.pyplot as plt

from dynamic_systems import LinearSystem
from controllers.compatibility import MatrixPencil, QFunction
from common import hessian_quadratic, vector2sym, rot2D, randomR, genStableLTI

n, m = 2, 2

# A, B = genStableLTI(n, m, type='float', Alims=(-2, 2), Blims=(-2, 2), place=True)
# A = np.zeros((n,n))
# B = np.eye(n)

A = np.array([[-2, 0],
              [ 1,-1]],dtype=float)

B = np.array([[2, 0],
              [1, 1]],dtype=float)

A += .1*np.random.randn(n,n)
B += .1*np.random.randn(n,m)

rankC = np.linalg.matrix_rank( control.ctrb(A,B) )
if rankC < n:
    print("System is not controllable.")
else:
    print("System is controllable.")

G = (B @ B.T)

plant = LinearSystem(initial_state=np.zeros(n), initial_control=np.zeros(n), A=A, B=B)

''' ---------------------------- Define quadratic CLF and CBF ----------------------------------- '''

# CLFeigs = np.random.randint(low=1, high=10, size=n)
# CLFcenter = np.zeros(n)
# Hv = hessian_quadratic(CLFeigs, randomR(n) )

# CBFeigs = np.random.randint(low=1, high=10, size=n)
# CBFcenter = 5*np.random.randn(n)
# Hh = hessian_quadratic(CBFeigs, randomR(n) )

CLFeigs = np.array([ 20.0, 1.0 ])
CLFcenter = np.array([ 0.0, 0.0 ])
rotCLF = rot2D(np.deg2rad(0))
Hv = hessian_quadratic(CLFeigs, rotCLF )

CBFeigs = np.array([ 1.0, 4.0 ])
CBFcenter = np.array([ 0.0, 3.0 ])
rotCBF = rot2D(np.deg2rad(80.0))
Hh = hessian_quadratic(CBFeigs, rotCBF )

p = 1.0

''' ---------------------------- Compute pencil and q-function ----------------------------------- '''
M = G @ Hh
N = p * G @ Hv - A
w = N @ ( CBFcenter - CLFcenter )

pencil = MatrixPencil(M, N)
for k, eig in enumerate( pencil.eigens ):
    print(f"{k+1}-th gen. eigenvalue of P(λ) = {eig.eigenvalue}.")

qfun = QFunction(pencil, Hh, w)
for k, eig in enumerate( qfun.Smatrix_pencil.eigens ):
    print(f"{k+1}-th gen. eigenvalue of S(λ) = {eig.eigenvalue}")

''' --------------------------------- Test compatibilization ------------------------------------- '''
def Hvfun(var):
    eps = 0.01
    L = vector2sym(var)
    return L @ L.T + eps * np.eye(n)

clf_dict = {"Hv_fun": Hvfun, "center": CLFcenter, "Hv": Hv }

''' ------------------------------------ Plot ----------------------------------- '''

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
fig.suptitle('Q-function plot')

# Plot before compatibilization
qfun.plot(ax)

# results = qfun.compatibilize( plant, clf_dict, p=p )

# newHv = results["Hv"]
# final_cost = results["cost"]
# compatibility = results["compatibility"]

# print("Results of compatibilization process are:")
# print(f"Original Hv = \n{Hv}, \nCompatible Hv = \n{newHv}")
# print(f"Final cost {final_cost}, compatibility eigenvalues = {compatibility}")

# qfun.plot(ax[1])

plt.show()