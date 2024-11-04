import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from controllers.compatibility import MatrixPencil

n, p = (2, 2)

M = np.random.randn(n,p)
M = M.T @ M/10

N = np.random.randn(n,p)
N = N.T @ N/100

H = np.random.randn(p,p)
H = H.T @ H/100    # H is usually psd
w = np.random.randn(n)

pencil = MatrixPencil(M, N)
pencil_poly = pencil.get_poly()

print("Testing pencil λ M - N, with")
print(f"M = \n{M}, \nN = \n{N}")
print(f"Matrix pencil polynomial = \n{pencil_poly}")

if pencil.type != 'singular':
    pencil.eigen()
    print("Pencil eigenvalues:")
    for k, eig in enumerate(pencil.eigens):
        print(f"λ{k+1} = {eig.eigenvalue}")
        for eigvec in eig.r_eigenvectors:
            print(f"z{k+1} = {eigvec}")
            print(f"|| (λ{k+1} M - N) z{k+1} || = {np.linalg.norm( pencil(eig.eigenvalue) @ eigvec )}")

    print(f"Real eigenvalues = {pencil.get_real_eigen()}")

null_poly = pencil.get_nullspace()
print(f"Null space Λ(λ) Z = \n{null_poly}")
print(f"Error (λ M - N) Λ(λ) Z = \n{pencil_poly @ null_poly}")

pencil.qfunction(H, w)
sols = pencil.equilibria()

for k, sol in enumerate(sols):
    l = sol["lambda"]
    s = sol["stability"]
    print(f"λ{k+1} = {l} with stability = {s}")

# ------------------------------------ Plot -----------------------------------
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.0, 5.0), layout="constrained")
fig.suptitle('Test General Matrix Pencil')

pencil.plot_qfunction(ax)

plt.show()