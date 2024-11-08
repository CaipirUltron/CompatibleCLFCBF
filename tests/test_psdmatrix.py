import sys
import control
import numpy as np

def genNum(min=0,max=10):
    return np.random.randint(min, max)

n = 3
minimum, maximum = 0, 10

numTests = 1000
for _ in range(numTests):

    G = np.zeros((n,n))
    for _ in range(n): 
        sigma = np.random.randint(0,10)
        x = np.random.randn(n)
        G += sigma* np.outer(x,x)

    A = np.random.randn(n,n)
    B = np.random.randn(n,n)

    # Creates desired poles based on number of states
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
    P = np.linalg.inv(A - B @ K)
    PG = P @ G
    # print(f"Eigs of P = {np.linalg.eigvals(P)}")

    eigs = np.linalg.eigvals(PG + PG.T)
    if np.any(eigs.real < -1e-12):
        print(f"Conjecture is wrong. Counter example eig. is: {eigs}, for {desired_poles}")
        sys.exit()

print(f"Conjecture most likely correct after {numTests} trials")