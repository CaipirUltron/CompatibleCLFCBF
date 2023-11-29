import numpy as np
from controllers import LinearMatrixPencil
import cvxpy as cp

dim = 6

Q = np.random.rand(dim, dim)
L = np.random.rand(dim, dim)

var = cp.Variable((dim, dim))
param = cp.Parameter((dim, dim))

objective = cp.Minimize( cp.norm( var - param , "fro") )
constraint = [ var + var.T >> 0 ]
problem = cp.Problem(objective, constraint)

param.value = L
problem.solve(verbose=True)
L = var.value

eigsL = np.linalg.eigvals(L)

print("Eigenvalues of L = " + str(eigsL))

# Constructs pencil with a non-symmetric but p.s.d. matrix
pencil = LinearMatrixPencil(Q @ Q.T, L+L.T)