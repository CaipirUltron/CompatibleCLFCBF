import sys
import control
import numpy as np
from controllers.compatibility import MatrixPencil

def genNum(min=0,max=10):
    return np.random.randint(min, max)

n, m = 3, 3
minimum, maximum = 0, 10

numTests = 1000
for i in range(numTests):

    # G = np.zeros((n,n))
    # for _ in range(n): 
    #     sigma = np.random.randint(0,10)
    #     x = np.random.randn(n)
    #     G += sigma* np.outer(x,x)

    # Creates desired poles based on number of states
    A = np.random.randn(n,n)
    B = np.random.randn(n,m)

    G = B @ B.T

    Hh = np.random.randn(n,n)
    Hh = Hh.T @ Hh
    M = G @ Hh

    Hv = np.random.randn(n,n)
    Hv = Hv.T @ Hv
    N = G @ Hv

    real_part = np.zeros(n)
    imag_part = np.zeros(n, dtype='complex')
    for k in range(int(np.floor(n/2))):
        real = (maximum-minimum)*np.random.rand() + minimum
        imag = (maximum-minimum)*np.random.rand() + minimum
        real_part[2*k:2*k+2] = np.array([ real, real ])
        imag_part[2*k:2*k+2] = np.array([ imag*(1j), -imag*(1j) ])
    if n%2 != 0: real_part[-1] = (maximum-minimum)*np.random.rand() + minimum

    desired_poles = real_part + imag_part
    K = control.place(A, B, p=desired_poles)
    Acl = A - B @ K
    N -= Acl

    pencil = MatrixPencil(M, N)
    stability_pencil = MatrixPencil(M + M.T, N + N.T)

    Min, Max = 1, 10
    l = (Max - Min)*np.random.randn() - Min
    P = pencil(l)
    eigP = np.linalg.eigvals(P)
    try:
        Pinv = np.linalg.inv(P)
    except:
        continue

    PinvG = Pinv @ G
    Sym_PinvG = PinvG + PinvG.T
    eigPinvG = np.linalg.eigvals(Sym_PinvG)

    Pinv1 = Pinv @ M @ Pinv @ G
    Q1 = Pinv1 + Pinv1.T
    eigQ1 = np.linalg.eigvals(Q1)

    Pinv2 = Pinv @ M @ Pinv
    Q2 = Pinv2 + Pinv2.T
    eigQ2 = np.linalg.eigvals(Q2)

    # fail = np.all( eigQ1.real > 0 ) and np.any( eigQ2.real < 0 )
    fail = np.any( stability_pencil.get_real_eigen() < 0 )
    if fail:
        print(f"Conjecture is wrong after {i} trials. Counter example is:")
        print(f"Q eigs = {eigQ1}")
        sys.exit()

print(f"Conjecture most likely correct after {numTests} trials")