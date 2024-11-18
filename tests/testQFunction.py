import numpy as np
import matplotlib.pyplot as plt

from common import randomGen, genStableLI, hessian_2Dquadratic
from controllers import MatrixPencil, QFunction

n, m = 2, 2

while True:

    A, B = genStableLI(n, m, type='int', random_lim=(-10, +10), real_lim=(-10, -1), imag_lim=(0, 1))
    G = (B @ B.T)

    CLFeigs = np.random.randint(low=1, high=10, size=n)
    CLFangle = np.deg2rad( np.random.randint(low=-180,high=180 ) )
    CLFcenter = 10*np.random.randn(2)
    Hv = hessian_2Dquadratic(CLFeigs, CLFangle)

    CBFeigs = np.random.randint(low=1, high=10, size=n)
    CBFangle = np.deg2rad( np.random.randint(low=-180,high=180 ) )
    CBFcenter = 10*np.random.randn(2)
    Hh = hessian_2Dquadratic(CBFeigs, CBFangle)

    w = np.random.randn(n)

    M = G @ Hh
    N = G @ Hv - A

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

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 5.0), layout="constrained")
    fig.suptitle('Q-function')
    qfun.plot(ax)

    # plt.show()

    # Keeps generating new figures
    plt.pause(1e-3)
    plt.ginput(timeout=0)
    plt.close(fig)