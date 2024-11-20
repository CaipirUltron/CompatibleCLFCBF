import numpy as np
from numpy.polynomial import Polynomial as Poly

import matplotlib.pyplot as plt

from common import randomR, genStableLI, hessian_quadratic
from controllers import MatrixPencil, QFunction, solve_poly_orthonormal

n, m = 3, 3

loop = False
# loop = True

v = np.array([ Poly([1,2,3,4,-1]), Poly([-5,4,0,3,1]), Poly([1,0,-2,7,2]) ])
print(v)
solve_poly_orthonormal(v)

def generateAndPlot(ax):
    '''
    Generates new example and plot it into ax
    '''
    A, B = genStableLI(n, m, stabilize=False, type='int', random_lim=(-10, +10), real_lim=(-10, -1), imag_lim=(0, 1))
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
    pencilEigens = pencil.get_eigen()
    for k, eig in enumerate(pencilEigens):
        print(f"{k+1}-th gen. eigenvalue = {eig.eigenvalue}.")

    if pencil.has_real_spectra():
        print("The given matrix pencil has real spectra.")

        stabilityEigens = pencil.symmetric().get_real_eigen()
        for k, eig in enumerate(stabilityEigens):
            text = f"{k+1}-th symmetric eigenvalue = {eig.eigenvalue} is of the "
            if eig.inertia > 0.0:
                text += "positive"
            else:
                text += "negative"
            print(text+" type.")

    qfun = QFunction(pencil, Hh, w)
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