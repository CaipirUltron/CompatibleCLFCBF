import control as cs
import numpy as np
import scipy as sp
from numpy.linalg import eigvals as eigs

n = 3

G = np.random.randn(n,n)
H1 = np.random.randn(n,n)
H2 = np.random.randn(n,n)

A = np.random.randn(n,n)
B = np.random.randn(n,n)

K = cs.place(A, B, p=[ -1, -2, -3 ])
Acl = A - B @ K

G = G.T @ G
H1 = H1.T @ H1
H2 = H2.T @ H2

M = G @ H1
N = G @ H2 - Acl

print(f"Eigenvalues of M = {eigs(M)}")
print(f"Eigenvalues of N = {eigs(N)}")

MM, NN, alpha, beta, Q, Z = sp.linalg.ordqz(N, M)

eigenvalues = alpha/beta
print(f"Polar gen. eigenvalues of λ M - N =")
for k,(a,b) in enumerate(zip(alpha, beta)):
    print(f"{k+1}-eigen = ({a},{b})")
print(f"Gen. eigenvalues of λ M - N = {eigenvalues}")

