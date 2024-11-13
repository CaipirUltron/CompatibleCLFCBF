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
        # imag_part[2*k:2*k+2] = np.array([ imag*(1j), -imag*(1j) ])
    if n%2 != 0: real_part[-1] = (maximum-minimum)*np.random.rand() + minimum

    desired_poles = real_part + imag_part
    K = control.place(A, B, p=desired_poles)
    Acl = A - B @ K
    N -= Acl

    pencil = MatrixPencil(M, N)
    real_eigP = pencil.get_real_eigen()
    if len(real_eigP) > 2:
        Min, Max = real_eigP[0], real_eigP[1]
        delta = Max - Min
        l = delta/2 + Min
    else:
        continue

    P = pencil(l)
    eigP = np.linalg.eigvals(P)
    try:
        Pinv = np.linalg.inv(P)
    except:
        continue

    PinvG = Pinv @ G
    Sym_PinvG = PinvG + PinvG.T
    eigPinvG = np.linalg.eigvals(Sym_PinvG)

    PinvG2 = PinvG @ Hh @ PinvG
    Sym_PinvG2 = PinvG2 + PinvG2.T
    eigPinvG2 = np.linalg.eigvals(Sym_PinvG2)

    fail = np.any( eigPinvG2.real < 0)

    if fail:
        print(f"Conjecture is wrong after {i} trials. Counter example is:")
        print(f"Q eigs = {eigPinvG2}")
        sys.exit()

print(f"Conjecture most likely correct after {numTests} trials")