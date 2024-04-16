import numpy as np
import scipy as sp
from scipy.optimize import minimize

from common import *
from functions import Kernel

def var_to_PSD(var: np.ndarray) -> np.ndarray:
    '''Transforms an n(n+1)/2 array representing a stacked symmetric matrix into standard PSD form'''
    sqrtP = vector2sym(var)
    P = sqrtP.T @ sqrtP
    return P

def PSD_to_var(P: np.ndarray) -> np.ndarray:
    '''Transforms a standard PSD matrix P into an array of size n(n+1)/2 list representing the stacked symmetric square root matrix of P'''
    return sym2vector(sp.linalg.sqrtm(P))

d=3
kernel = Kernel(dim=2, degree=d)
kernel_dim = kernel._num_monomials
As = kernel.Asum

Pnom = np.random.randn(kernel_dim, kernel_dim)
Pnom = Pnom.T @ Pnom

def objective(var: np.ndarray) -> float:
    ''' Minimizes the changes to the CLF geometry needed for compatibilization '''
    P = var_to_PSD(var)
    cost = np.linalg.norm( P - Pnom, 'fro')
    return cost

def constraint(var: np.ndarray) -> float:
    ''' Constraint on lowerbound matrix '''

    P = var_to_PSD(var)

    L = lyap( (As).T, lyap( (As).T, P ) )
    # L = lyap( (As @ As).T, P )

    return np.min( np.linalg.eigvals(L) )

constraints = [ {"type": "ineq", "fun": constraint} ]
init_var = PSD_to_var(Pnom)
sol = minimize( objective, init_var, constraints=constraints )

P = var_to_PSD( sol.x )
R = lyap( (As).T, lyap( (As).T, P ) )

print(f"λ(P) = {np.linalg.eigvals(P)}")
print(f"λ(R) = {np.linalg.eigvals(R)}")