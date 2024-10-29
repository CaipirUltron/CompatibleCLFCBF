import control
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings

from itertools import product
from common import hessian_2Dquadratic
from numpy.polynomial import Polynomial as Poly
from controllers.compatibility import MatrixPencil

n = 2
m = 1

A = np.random.randn(n,n)
B = np.random.randn(n,m)

BB = (B @ B.T)

real = np.random.randint(-10, -1)
imag = np.random.randint(0, 100)
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

''' ---------------------------- Define quadratic CLF and CBF ----------------------------------- '''

CLFeigs = np.array([ 100.0, 1.0 ])
CLFangle = np.deg2rad(2)
CLFcenter = np.array([0.0, 0.0])
Hv = hessian_2Dquadratic(CLFeigs, CLFangle)

CBFeigs = np.array([ 1.0, 1.0 ])
CBFangle = np.deg2rad(0)
CBFcenter = np.array([0.0, 5.0])
Hh = hessian_2Dquadratic(CBFeigs, CBFangle)

p = 1.0
''' ---------------------------- Compute pencil and q-function ----------------------------------- '''

M = BB @ Hh 
N = p * BB @ Hv - Acl
w = N @ CBFcenter - p * BB @ Hv @ CLFcenter

pencil = MatrixPencil(M,N)
print(f"Pencil λ M - N spectra = {pencil.eigenvalues}")

pencil.qfunction(Hh, w)

''' ------------------------------------ Plot ----------------------------------- '''
n_poly, d_poly = pencil.get_qfunction()
print(f"n(λ) = {n_poly}")
print(f"d(λ) = {d_poly}")

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.0, 5.0), layout="constrained")
fig.suptitle('Test Linear System Pencil')

pencil.plot_qfunction(ax, res=0.1)
plt.show()