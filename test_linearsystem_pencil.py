import control
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings

from common import hessian_2Dquadratic

n = 2
m = 2

A = np.random.randn(n,n)
B = np.random.randn(n,m)

BB = (B @ B.T)

real = np.random.randint(-10, -1)
imag = np.random.randint(0, 10)
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
MM, NN, alpha, beta, Q, Z = sp.linalg.ordqz(N, M)
w = N @ CBFcenter - p * BB @ Hv @ CLFcenter

print(f"MM = {MM}")
print(f"NN = {NN}")

print("Pencil is of the form λ M - N")
print(f"Spectra of M = {np.linalg.eigvals(M)}")
print(f"Spectra of N = {np.linalg.eigvals(N)}")
print(f"Polar spectra of λ M - N = ")
for a, b in zip(alpha, beta):
    print(f"({a},{b})")

pencil_eigs = alpha/beta
if np.linalg.norm(pencil_eigs.imag) > 1e-12:
    warnings.warn("Pencil has imaginary eigenvalues.")
else:
    pencil_eigs = pencil_eigs.real
    if np.any(pencil_eigs <= 0.0):
        warnings.warn("Pencil has non-positive eigenvalues.")
    pencil_eigs = np.sort(pencil_eigs)

print(f"Pencil λ M - N spectra = {pencil_eigs}")

def pencil(M: np.ndarray, N: np.ndarray, l: float):
    ''' 
    Linear matrix pencil of the form l * M - N
    '''
    if not isinstance(l, (int, float)):
        raise ValueError("The lambda value must be a real number.")

    if M.shape != N.shape:
        raise TypeError("The shapes of A and B must be the same.")    

    return l * M - N

def q_function(lambdas: np.ndarray):
    ''' 
    Compute q function for a given array of lambda values 
    '''
    q = np.zeros(len(lambdas))
    positive_interval, solutions = [], []
    for k, l in enumerate(lambdas):

        P = pencil(M, N, l)

        ''' Checks if P is definite on some interval '''
        eigsP = np.linalg.eigvals( P )
        eigsP = eigsP.real
        eigsP = np.sort(eigsP)
        if np.all( eigsP > 0.0 ):
            positive_interval.append(l)

        try:
            v = np.linalg.inv(P) @ w
            grad_hi = Hh @ v
            q[k] = v.T @ Hh @ v
        except Exception as error:
            if isinstance(error, np.linalg.LinAlgError):
                q[k] = np.inf
            
        if np.abs(q[k] - 1.0) <= 1e-3:
            perp = np.array([ grad_hi[1], -grad_hi[0] ])
            solutions.append({"lambda": l, "stability": perp.T @ P @ perp})

    return q, positive_interval, solutions

''' ------------------------------------ Plot ----------------------------------- '''

lambda_min = 0
if pencil_eigs[-1] < np.inf:
    lambda_max = 5*pencil_eigs[-1]
else:
    lambda_max = 1000
lambda_res = 0.01
lambdas = np.arange(lambda_min, lambda_max, lambda_res)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.0, 5.0), layout="constrained")
fig.suptitle('Test Linear System Pencil')

for k in range(n):
    if np.abs( pencil_eigs[k].imag ) < 1e-12:
        ax.plot( [ pencil_eigs[k] for l in lambdas ], np.linspace(0, 100, len(lambdas)), 'b--' )

q, positive_interval, sols = q_function(lambdas)
if len(positive_interval) > 0:
    print(f"Positive interval of λ M - N = [{positive_interval[0]},{positive_interval[-1]}]")

stable_pts, unstable_pts = [], []
for sol in sols:
    l = sol["lambda"]
    stability = sol["stability"]
    if stability > 0:
        unstable_pts.append(l)
    else:
        stable_pts.append(l)

ax.plot( stable_pts, [ 1.0 for s in stable_pts], 'ro' )
ax.plot( unstable_pts, [ 1.0 for s in unstable_pts], 'bo' )
ax.plot( lambdas, [ 1.0 for l in lambdas ], 'r--' )
ax.plot( lambdas, q, label='q' )
ax.set_xlim(lambda_min, lambda_max) 
ax.set_ylim(0, 100) 

plt.show()