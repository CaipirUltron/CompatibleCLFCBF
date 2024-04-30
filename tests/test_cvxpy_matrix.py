import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from common import lyap, symmetric_basis

p = 6
I = np.eye(p)

P_var = cp.Variable( (p,p), symmetric=True )
R_var = cp.Variable( (p,p), symmetric=True )
Pnom_var = cp.Parameter( (p,p), symmetric=True )

A = np.zeros((p,p))
for basis_vec in symmetric_basis(p):
    A += np.random.randn() * basis_vec

kron_matrix = ( np.kron(I,A.T) + np.kron(A, I) )
kron_matrix = kron_matrix @ kron_matrix

cost = cp.norm(P_var - Pnom_var)
constraints = [ P_var >> 0
               , kron_matrix @ cp.vec(P_var) == cp.vec(R_var)
               , lyap(A.T, lyap(A.T, P_var)) >> 0 ]
problem = cp.Problem( cp.Minimize(cost), constraints )

Pnom = np.random.rand(p,p)
Pnom_var.value = Pnom.T @ Pnom
problem.solve(verbose=True, max_iters=50000)

print(f"λ(A) = {np.linalg.eigvals(A)}")
print(f"Eigenvals of P:\n {np.linalg.eigvals(P_var.value)}")
print(f"Eigenvals of R:\n {np.linalg.eigvals(lyap(A.T, lyap(A.T, P_var.value)))}")
