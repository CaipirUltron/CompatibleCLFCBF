import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve, least_squares
from common import rot2D
from functions import Kernel
from numpy.polynomial import Polynomial as Poly

# np.set_printoptions(precision=3, suppress=True)

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Hyperbolic Expansion")
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

N = 1000
t_list = np.linspace(0, 100, N)

# ---------------- Real Hyperbola -------------
# axis = [ 1, 1 ]
# center = [ 2, 2 ]
# angle = 40
# R = rot2D( np.deg2rad(angle) )
# const = 20

# ''' First half '''
# x1_list = axis[0]*np.cosh(t_list - const) 
# y1_list = axis[1]*np.sinh(t_list - const)
# x1_rot = [ R[0,:].T @ np.array([x,y]) + center[0] for x,y in zip(x1_list, y1_list) ]
# y1_rot = [ R[1,:].T @ np.array([x,y]) + center[1] for x,y in zip(x1_list, y1_list) ]
# ax.plot(x1_rot, y1_rot, 'b')

# ''' Second half '''
# x2_list = -x1_list
# y2_list = y1_list
# x2_rot = [ R[0,:].T @ np.array([x,y]) + center[0] for x,y in zip(x2_list, y2_list) ]
# y2_rot = [ R[1,:].T @ np.array([x,y]) + center[1] for x,y in zip(x2_list, y2_list) ]
# ax.plot(x2_rot, y2_rot, 'b')

# ---------------- Approx. Hyperbola -------------
# order = 10

# ''' First half approx. '''
# num_samples = len(t_list)
# sum_cosh, sum_sinh = np.zeros(num_samples), np.zeros(num_samples)
# for k, t in enumerate(t_list):
#     sum_cosh[k] = np.sum([ (1/math.factorial(2*n))*((t-const)**(2*n)) for n in range(order) ])
#     sum_sinh[k] = np.sum([ (1/math.factorial(2*n+1))*((t-const)**(2*n+1)) for n in range(order) ])

# print(f"cosh = {sum_cosh[0]}")
# print(f"sinh = {sum_sinh[0]}")

# x1_list = axis[0]*sum_cosh
# y1_list = axis[1]*sum_sinh

# x1_rot = [ R[0,:].T @ np.array([x,y]) + center[0] for x,y in zip(x1_list, y1_list) ]
# y1_rot = [ R[1,:].T @ np.array([x,y]) + center[1] for x,y in zip(x1_list, y1_list) ]
# ax.plot(x1_rot, y1_rot, 'r--')

# ''' Second half approx. '''
# x2_list = -x1_list
# y2_list = y1_list
# x2_rot = [ R[0,:].T @ np.array([x,y]) + center[0] for x,y in zip(x2_list, y2_list) ]
# y2_rot = [ R[1,:].T @ np.array([x,y]) + center[1] for x,y in zip(x2_list, y2_list) ]
# ax.plot(x2_rot, y2_rot, 'r--')

# plt.show()

# ---------------- Function for invariant set estimation ---------------
def fit_invariant(A, B, kernel, order):
    '''
    This function fits the parameters of a polynomial of the type xi = Σ [pi]_k t**k,
    of maximum order order, in order to solve the equation (t A - B) m(x) = 0, where
    m(x) is the kernel function of x.
    '''
    if A.shape != B.shape:
        raise Exception("Matrix sizes are not compatible.")
    n = A.shape[0]
    p = A.shape[1]
    
    if len(kernel) != p:
        raise Exception("Kernel and matrix sizes are not compatible.")
    
    ''' 
    P is the parameter matrix to be fitted. 
    Each line represents the parameter for each state xi.
    Each column represents the corresponding power of t, from 0 to kappa. 
    '''
    kernel_degrees = [ sum(alpha) for alpha in kernel ]
    max_degree = order*max(kernel_degrees) + 1
    polyAB = [[ Poly([ -B[i,j], A[i,j] ]) for j in range(p) ] for i in range(n) ]

    def f(params):
        ''' Objective function '''
        P = params.reshape((n, order+1))

        ''' List of polynomials representing the solution for each state '''
        x_poly = [ Poly(P[i,:]) for i in range(n) ]

        f = np.zeros(n*(max_degree+1))
        for i in range(n):
            poly_line = Poly(0.0)
            for j, alpha in enumerate(kernel):        
                mj_poly = np.prod([ x_poly[dim]**alpha[dim] for dim in range(n) ])
                poly_line += polyAB[i][j] * mj_poly
            poly_line.coef = np.hstack([ poly_line.coef, np.zeros( max_degree - poly_line.degree() ) ])
            f[(max_degree+1)*i:(max_degree+1)*(i+1)] = poly_line.coef

        return f

    P0 = np.random.randn(n, order+1)
    param0 = P0.flatten()
    param = least_squares( f, param0 )
    print(param)
    P = param.x.reshape((n, order+1))

    return P

# ------------------------------------ Test invariant fitting -----------------------------------
n, d = 2, 1
kernel = Kernel(dim=n, degree=d)
print(kernel)
kernel_powers = kernel._powers
print(kernel_powers)

clf_eigen = [1, 1]
clf_angle = 20
clf_center = np.array([0.1, 0.1]).reshape((2,1))

cbf_eigen = [1, 1]
cbf_angle = 40
cbf_center = np.array([0.1, 4]).reshape((2,1))

Rv = rot2D( np.deg2rad(clf_angle) )
Hv = Rv @ np.diag(clf_eigen) @ Rv.T
B = Hv @ np.hstack([ -clf_center, np.eye(n) ])

Rh = rot2D( np.deg2rad(cbf_angle) )
Hh = Rh @ np.diag(cbf_eigen) @ Rh.T
A = Hh @ np.hstack([ -cbf_center, np.eye(n) ])

order = 4
P = fit_invariant(A, B, kernel_powers, order=order)

print(P)

x_list = np.sum([ P[0,i] * (t_list**i) for i in range(order+1) ])
y_list = np.sum([ P[1,i] * (t_list**i) for i in range(order+1) ])

ax.plot(x_list, y_list, 'r--')

plt.show()