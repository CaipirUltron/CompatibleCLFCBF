import numpy as np
import matplotlib.pyplot as plt

from common import randomR, genStableLI, hessian_quadratic
from controllers import MatrixPencil, QFunction

n, m = 3, 3

loop = False
# loop = True

def generateAndPlot(ax):
    '''
    Generates new example and plot it into ax
    '''
    # A, B = genStableLI(n, m, stabilize=False, type='int', random_lim=(-10, +10), real_lim=(-10, -1), imag_lim=(0, 1))
    A, B = genStableLI(n, m, stabilize=False, type='float')

    G = (B @ B.T)

    CLFeigs = np.random.randint(low=1, high=10, size=n)
    CLFcenter = np.random.randn(n)
    Hv = hessian_quadratic(CLFeigs, randomR(n) )

    CBFeigs = np.random.randint(low=1, high=10, size=n)
    CBFcenter = np.random.randn(n)
    Hh = hessian_quadratic(CBFeigs, randomR(n) )

    M = G @ Hh
    N = G @ Hv - A

    w = N @ ( CBFcenter - CLFcenter )

    pencil = MatrixPencil(M, N)
    for k, eig in enumerate(pencil.eigens):
        print(f"{k+1}-th gen. eigenvalue = {eig.eigenvalue}.")

    qfun = QFunction(pencil, Hh, w)
    for k, eig in enumerate( qfun.stability_pencil.real_eigen() ):
        print(f"{k+1}-th real eigenvalue of S(λ) companion form = {eig.eigenvalue}")

    qfun.compatibility_matrix._test_sos_decomposition(100)
    qfun.plot(ax)

if not loop:
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
    fig.suptitle('Q-function')
    generateAndPlot(ax)
    plt.show()

while loop:

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
    fig.suptitle('Q-function')

    generateAndPlot(ax)
    
    plt.pause(1e-3)
    plt.ginput(timeout=0)
    plt.close(fig)