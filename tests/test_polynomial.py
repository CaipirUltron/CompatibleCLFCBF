'''
Test polynomial functions
'''
import numpy as np
from functions import PolynomialFunction
from common import *

n = 2
max_degree = 3

# P = np.random.rand(p,p)
# P = P.T @ P
# eigP , _= np.linalg.eig(P)
# P = P/np.max(eigP)

state = np.random.rand(n)
clf = PolynomialFunction(*state, degree = max_degree)

print(clf.get_kernel())
print( generate_monomial_list(n, max_degree) )