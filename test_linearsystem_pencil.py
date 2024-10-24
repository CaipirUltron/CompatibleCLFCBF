import control
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings

from itertools import product
from common import hessian_2Dquadratic
from numpy.polynomial import Polynomial as Poly

n = 2
m = 1

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
MM, NN, diagMM, diagNN, Q, Z = sp.linalg.ordqz(M, N, output='complex')
w = N @ CBFcenter - p * BB @ Hv @ CLFcenter

print(f"M error = {np.linalg.norm(Q @ MM @ Z.conjugate().T - M)}")
print(f"N error = {np.linalg.norm(Q @ NN @ Z.conjugate().T - N)}")

print("Pencil is of the form λ M - N")
print(f"Spectra of M = {np.linalg.eigvals(M)}")
print(f"Spectra of N = {np.linalg.eigvals(N)}")

pencil_eigs = diagNN / diagMM
if np.linalg.norm(pencil_eigs.imag) > 1e-12:
    warnings.warn("Pencil has imaginary eigenvalues.")
else:
    pencil_eigs = pencil_eigs.real
    if np.any(pencil_eigs <= 0.0):
        warnings.warn("Pencil has non-positive eigenvalues.")
    pencil_eigs = np.sort(pencil_eigs)

print(f"Pencil λ M - N spectra = {pencil_eigs}")

def solve_poly_linearsys(T: np.ndarray, S: np.ndarray, b_poly: np.ndarray) -> np.ndarray:
    '''
    Finds the polynomial array x(λ) tha solves (λ T - S) x(λ) = b(λ), where T, S are 1x1 or 2x2
    and b(λ) is a polynomial array of arbitrary order.
    '''
    if isinstance(T, (int, float)): T = np.array([[T]])
    if isinstance(S, (int, float)): S = np.array([[S]])

    if T.shape != S.shape:
        raise TypeError("T and S must have the same shape.")
    
    if T.shape[0] != T.shape[0]:
        raise TypeError("T and S must be square matrices.")

    blk_size = T.shape[0]
    bshape = b_poly.shape
    if bshape[0] != blk_size:
        raise TypeError("Number of lines in (λ T - S) and b(λ) must be the same.")

    # Extract arrays from b_poly and store in b_coefs list (variable size)
    bsys = np.zeros((0, bshape[1]))
    for (i,j), poly in np.ndenumerate(b_poly):

        if not isinstance( poly, Poly ):
            raise TypeError("b(λ) is not an array of polynomials.")
        
        # Setup bsys
        b_order = len(poly.coef)
        n_coefs_toadd = b_order - int(bsys.shape[0] / blk_size)
        if n_coefs_toadd > 0:
            bsys = np.vstack([ bsys ] + [ np.zeros((blk_size, bshape[1])) for _ in range(n_coefs_toadd) ])

        for k, c in enumerate(poly.coef):
            bsys[ k * blk_size + i, j ] = c

    # Constructs the Asys and bsys matrices
    b_order = int(bsys.shape[0] / blk_size)
    Asys = np.zeros([ b_order*blk_size, (b_order-1)*blk_size ])
    for i in range(b_order-1):
        Asys[ i*blk_size:(i+1)*blk_size , i*blk_size:(i+1)*blk_size ] = -S
        Asys[ (i+1)*blk_size:(i+2)*blk_size , i*blk_size:(i+1)*blk_size ] = T

    results = np.linalg.lstsq(Asys, bsys)
    x_coefs = results[0]
    residuals = results[1]
    print(f"Residuals norm = {np.linalg.norm(residuals)}")
    # print(f"x_coefs = {x_coefs}")

    x_poly = np.array([[ Poly([0.0 for _ in range(b_order-1) ]) for j in range(bshape[1]) ] for i in range(blk_size) ])
    for (i,j), c in np.ndenumerate(x_coefs):
        exp = int(i/blk_size)
        x_poly[i%blk_size,j].coef[exp] = c

    return x_poly

def compute_qfunction_poly(M: np.ndarray, N: np.ndarray, Hh: np.ndarray, w: np.ndarray):
    '''
    Returns the numerator and denominator polynomials n(λ), d(λ) of the q-function q(λ) = n(λ)/d(λ),
    where P(λ) v(λ) = w, q(λ) = v(λ).T @ Hh v(λ).
    '''
    equal_dims = [ M.shape == N.shape, M.shape == Hh.shape ]
    if not equal_dims:
        raise TypeError("M, N and Hh must have the same dimensions.")
    
    if M.shape[0] != M.shape[1]:
        raise TypeError("M, N and Hh must be a square matrices.")
    
    n = M.shape[0]
    if len(w) != n:
        raise TypeError("Vector w must have the same dimensions as the pencil λ M - N.")

    ''' Computes QZ decomposition of pencil '''
    MM, NN, diagMM, diagNN, Q, Z = sp.linalg.ordqz(M, N, output='real')

    ''' -------------------- Computes the block dimensions, poles and adjoints (TO BE USED later) ------------------------- '''
    blk_dims, blk_poles, blk_adjs = [], [], []
    for i in range(n):

        # 2X2 BLOCKS OF COMPLEX CONJUGATE PENCIL EIGENVALUES
        if i < n-1 and MM[i+1,i] != 0.0:
            blk_dims.append(2)

            MMblock = MM[i:i+2,i:i+2]
            NNblock = NN[i:i+2,i:i+2]

            a = np.linalg.det(MMblock)
            b = MMblock[0,0] * NNblock[1,1] + NNblock[0,0] * MMblock[1,1]
            c = np.linalg.det(NNblock)
            blk_poles.append( Poly([ c, b, a ]) )

            adj11 = Poly([ -NNblock[1,1],  MMblock[1,1] ])
            adj12 = Poly([           0.0, -MMblock[0,1] ])
            adj21 = Poly([           0.0, -MMblock[1,0] ])
            adj22 = Poly([ -NNblock[0,0],  MMblock[0,0] ])
            blk_adjs.append( np.array([[ adj11, adj12 ],[ adj21, adj22 ]]) )

        # 1X1 BLOCKS OF REAL PENCIL EIGENVALUES
        else:
            blk_dims.append(1)

            MMblock = MM[i,i]
            NNblock = NN[i,i]

            blk_poles.append( Poly([ -NNblock, MMblock ]) )
            blk_adjs.append( np.array([Poly(1.0)]) )
    
    ''' -------------------------- Computes the adjoint matrix --------------------------------- '''
    num_blks = len(blk_dims)
    adjoint = np.array([[ Poly([0.0]) for _ in range(n) ] for _ in range(n) ])
    for i in range(num_blks-1, -1, -1):
        blk_i_slice = slice( i*blk_dims[i], (i+1)*blk_dims[i] )

        for j in range(i, num_blks):
            blk_j_slice = slice( j*blk_dims[j], (j+1)*blk_dims[j] )

            # Computes ADJOINT DIAGONAL BLOCKS
            if j == i:
                poles_ij = np.array([[ np.prod([ pole for k, pole in enumerate(blk_poles) if k != i ]) ]])
                Lij = poles_ij @ blk_adjs[j]

            # Computes ADJOINT UPPER TRIANGULAR BLOCKS
            else:
                Tii = MM[ blk_i_slice, blk_i_slice ]
                Sii = NN[ blk_i_slice, blk_i_slice ]

                b_poly = np.array([[ Poly([0.0]) for _ in range(blk_dims[j]) ] for _ in range(blk_dims[i]) ])
                for k in range(i+1, j+1):
                    blk_k_slice = slice( k*blk_dims[k], (k+1)*blk_dims[k] )

                    # Compute polynomial (λ Tik - Sik) and get the kj slice of adjoint
                    Tik = MM[ blk_i_slice, blk_k_slice ]
                    Sik = NN[ blk_i_slice, blk_k_slice ]
                    poly_ik = np.array([[ Poly([ -Sik[a,b], Tik[a,b] ]) for b in range(Tik.shape[1]) ] for a in range(Tik.shape[0]) ])

                    adjoint_kj = adjoint[ blk_k_slice, blk_j_slice ]

                    b_poly -= poly_ik @ adjoint_kj

                Lij = solve_poly_linearsys( Tii, Sii, b_poly )

            # Populate adjoint matrix
            adjoint[ blk_i_slice, blk_j_slice ] = Lij

    ''' ---------------- Computes q-function numerator and denominator polynomials ------------------ '''
    Zinv = np.linalg.inv(Z)
    barHh = Zinv @ Hh @ (Zinv.T)
    barw = np.linalg.inv(Q) @ w

    n_poly = ( barw.T @ adjoint.T @ barHh @ adjoint @ barw )
    poles = np.prod([ pole for pole in blk_poles ])
    d_poly = poles**2

    return n_poly, d_poly

def pencil(M: np.ndarray, N: np.ndarray, l: float):
    ''' 
    Linear matrix pencil of the form λ M - N
    '''
    if not isinstance(l, (int, float)):
        raise ValueError("The lambda value must be a real number.")

    if M.shape != N.shape:
        raise TypeError("The shapes of A and B must be the same.")    

    return l * M - N

def q_function(lambdas: np.ndarray):
    ''' 
    Compute q function for a given array of λ values 
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

def q2_function(lambdas: np.ndarray, n_poly, d_poly):

    q = np.zeros(len(lambdas))
    for k,l in enumerate(lambdas):
        q[k] = n_poly(l) / d_poly(l)

    return q

''' ------------------------------------ Plot ----------------------------------- '''

def poly_qfunction_test(lambdas: np.ndarray):
    ''' Test rational q_function '''

    q1s = q_function(lambdas)[0]
    
    q2s = np.zeros(len(lambdas))
    for k, l in enumerate(lambdas):
        n, d = n_poly(l), d_poly(l)
        q2s[k] = n/d

    error = np.linalg.norm(q1s - q2s)
    print(f"q-function error = {error}")

n_poly, d_poly = compute_qfunction_poly(M, N, Hh, w)
print(f"n(λ) = {n_poly}")
print(f"d(λ) = {d_poly}")

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

q1, positive_interval, sols = q_function(lambdas)
q2 = q2_function(lambdas, n_poly, d_poly)

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
ax.plot( lambdas, q1, label='q1' )
ax.plot( lambdas, q2, '--', label='q2' )

ax.set_xlim(lambda_min, lambda_max) 
ax.set_ylim(0, 100) 
ax.legend()

plt.show()