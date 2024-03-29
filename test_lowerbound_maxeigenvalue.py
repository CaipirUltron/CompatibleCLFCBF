import numpy as np
import scipy as sp
from functions import Kernel
from common import symmetric_basis

n = 2
initial_state = [0.5, 6.0]

kernel = Kernel(*initial_state, degree=3)
p = kernel.kernel_dim
Amatrices = kernel.get_A_matrices()

N = 10000
for k in range(N):

    M = np.zeros([n,n])
    for B in symmetric_basis(n):
        M += np.random.randn()*B

    lambda_max = np.max( np.linalg.eigvals(M) )
    lambda_min = np.min( np.linalg.eigvals(M) )
    S = np.sum(M)

    print(f"λmax(M) = {lambda_max}")
    print(f"λmin(M) = {lambda_min}")
    print(f"lower bound on λmax(M) = {S/n}\n")

    hypothesis = lambda_max >= S/n

    if not hypothesis: 
        print("Hypothesis is false.")
        break

if hypothesis: print("Hypothesis is most likely true.")