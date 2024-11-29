import numpy as np
import matplotlib.pyplot as plt

from dynamic_systems import LinearSystem
from controllers.compatibility import MatrixPencil, QFunction
from common import hessian_quadratic, vector2sym, sym2vector, genStableLI, randomR

n, m = 2, 2

# A, B = genStableLI(n, m, type='int', random_lim=(-10, +10))
A, B = genStableLI(n, m, stabilize=False, type='float')
plant = LinearSystem(initial_state=np.zeros(n), initial_control=np.zeros(n), A=A, B=B)

G = (B @ B.T)

''' ---------------------------- Define quadratic CLF and CBF ----------------------------------- '''

CLFeigs = np.random.randint(low=1, high=10, size=n)
CLFcenter = np.random.randn(n)
Hv = hessian_quadratic(CLFeigs, randomR(n) )

CBFeigs = np.random.randint(low=1, high=10, size=n)
CBFcenter = np.random.randn(n)
Hh = hessian_quadratic(CBFeigs, randomR(n) )

p = 1.0

''' ---------------------------- Compute pencil and q-function ----------------------------------- '''

M = G @ Hh
N = p * G @ Hv - A
w = N @ ( CBFcenter - CLFcenter )

pencil = MatrixPencil(M, N)
for k, eig in enumerate(pencil.eigens):
    print(f"{k+1}-th gen. eigenvalue = {eig.eigenvalue}.")

qfun = QFunction(pencil, Hh, w)
for k, eig in enumerate( qfun.stability_pencil.real_eigen() ):
    print(f"{k+1}-th real eigenvalue of S(λ) companion form = {eig.eigenvalue}")

''' --------------------------------- Test compatibilization ------------------------------------- '''
def Hvfun(var):
    eps = 0.01
    L = vector2sym(var)
    return L @ L.T + eps * np.eye(n)

clf_dict = {"Hv_fun": Hvfun, "center": CLFcenter, "Hv": Hv }

''' ------------------------------------ Plot ----------------------------------- '''

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10.0, 5.0), layout="constrained")
fig.suptitle('Compatibilization of LTI System')

# Plot before compatibilization
qfun.plot(ax[0])

results = qfun.compatibilize( plant, clf_dict, p=p )

newHv = results["Hv"]
final_cost = results["cost"]
compatibility = results["compatibility"]

print("Results of compatibilization process are:")
print(f"Original Hv = \n{Hv}, \nCompatible Hv = \n{newHv}")
print(f"Final cost {final_cost}, compatibility eigenvalues = {compatibility}")

# Plot after compatibilization
qfun.plot(ax[1])

plt.show()