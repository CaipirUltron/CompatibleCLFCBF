import control
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from itertools import product
from numpy.linalg import eigvals as eigs

from numpy.polynomial import Polynomial as Poly

from common import hessian_2Dquadratic, vector2sym, sym2vector
from functions.multipoly import MultiPoly
from dynamic_systems import LinearSystem
from controllers.compatibility import MatrixPencil

n, m = 2, 2

A = np.random.randint(low=1, high=10, size=(n,n))
B = np.random.randint(low=1, high=10, size=(n,m))

BB = (B @ B.T)

real = np.random.randint(-10, -1)
imag = np.random.randint(0, 1)
# imag = 0.0
real_parts = np.array([ real, real ])
imag_parts = np.array([ imag*(1j), -imag*(1j) ])

rankC = np.linalg.matrix_rank( control.ctrb(A, B) )
if rankC < A.shape[0]:
    print(f"(A, B) pair is not controllable.")
else:
    print(f"(A, B) pair is controllable.")
    rankB = np.linalg.matrix_rank(B)
    if rankB >= rankC:
        desired_poles = real_parts + imag_parts
    else:
        desired_poles = [-1, -2]
    
    print(f"Desired poles = {desired_poles}")
    K = control.place(A, B, p=desired_poles)
    Acl = A - B @ K
    eigsAcl = np.linalg.eigvals(Acl)
    print(f"Poles of Acl = {eigsAcl}")

# Acl = np.zeros((n,n))
plant = LinearSystem(initial_state=np.zeros(n), initial_control=np.zeros(n), A=Acl, B=B)

''' ---------------------------- Define quadratic CLF and CBF ----------------------------------- '''

CLFeigs = np.array([ 6.0, 1.0 ])
CLFangle = np.deg2rad(0)
CLFcenter = np.array([0.0, 0.0])
Hv = hessian_2Dquadratic(CLFeigs, CLFangle)

CBFeigs = np.array([ 1.0, 4.0 ])
CBFangle = np.deg2rad(5)
CBFcenter = np.array([0.0, 5.0])
Hh = hessian_2Dquadratic(CBFeigs, CBFangle)

p = 1.0
''' ---------------------------- Compute pencil and q-function ----------------------------------- '''

M = BB @ Hh 
N = p * BB @ Hv - Acl
w = N @ CBFcenter - p * BB @ Hv @ CLFcenter

print(f"Hh = {Hh}")
print(f"Hv = {Hv}")

pencil = MatrixPencil(M,N)
pencil.eigen()
print(f"Pencil λ M - N spectra =")
for k, eig in enumerate(pencil.eigens):
    print(f"λ{k+1} = {eig.eigenvalue}")

pencil.qfunction(Hh, w)

''' --------------------------------- Test compatibilization ------------------------------------- '''
def Hvfun(var):
    eps = 0.01
    L = vector2sym(var)
    return L @ L.T + eps * np.eye(n)

clf_dict = {"Hv_fun": Hvfun, "center": CLFcenter, "Hv": Hv }
cbf_dict = {"Hh": Hh, "center": CBFcenter }

''' ------------------------------------ Plot ----------------------------------- '''
n_poly, d_poly = pencil.get_qfunction()
zero_poly = n_poly - d_poly

print(f"n(λ) = {n_poly}")
print(f"d(λ) = {d_poly}")
print(f"n(λ) - d(λ) = {zero_poly}")

n_poly = MultiPoly.from_nppoly( n_poly )
d_poly = MultiPoly.from_nppoly( d_poly )
zero_poly = MultiPoly.from_nppoly( zero_poly ) 

N = n_poly.sos_decomposition(verb=True)
D = d_poly.sos_decomposition(verb=True)

print(f"Error on SOS decomposition of n(λ) {n_poly.validate_sos()}")
print(f"Error on SOS decomposition of d(λ) {d_poly.validate_sos()}")

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10.0, 5.0), layout="constrained")
fig.suptitle('Compatibilization of Linear System')

pencil.plot_qfunction(ax[0], res=0.05)

Hv = pencil.compatibilize( plant, clf_dict, cbf_dict, p=1 )
pencil.plot_qfunction(ax[1], res=0.05)
plt.show()