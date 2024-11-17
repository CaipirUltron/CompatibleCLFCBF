import numpy as np
import matplotlib.pyplot as plt

from controllers import MatrixPencil, QFunction

n = 2

M = np.random.randn(n,n)
N = np.random.randn(n,n)

M = M.T @ M
N = N.T @ N

pencil = MatrixPencil(M, N)

H = np.random.randn(n,n)
H = H.T @ H
w = np.random.randn(n)
qfun = QFunction(pencil, H, w)

print("Starting QFunction unit tests.")

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.0, 5.0), layout="constrained")
fig.suptitle('Q-function')
qfun.plot(ax)
plt.show()

numTests = 100
for it in range(numTests):
    pass
    

print(f"Exit after testing with {it+1} random samples. \n")
# print(f"Pencil determinant error = {det_error}")