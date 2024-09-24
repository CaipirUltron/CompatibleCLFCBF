import math
import numpy as np
import matplotlib.pyplot as plt

from dynamic_systems import KernelAffineSystem
from scipy.optimize import fsolve, least_squares
from common import rot2D, kernel_quadratic
from functions import Kernel, KernelLyapunov, KernelBarrier, KernelFamily
from numpy.polynomial import Polynomial as Poly
from scipy.interpolate import approximate_taylor_polynomial

# np.set_printoptions(precision=3, suppress=True)
limits = 14*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Hyperbolic Expansion")
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(limits[0], limits[1])
ax.set_ylim(limits[2], limits[3])

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
def trim(poly):
    ''' Trims rational polynomial'''

    if np.all(poly == 0.0): return poly

    to_return = np.array([ True for _ in range(len(poly)) ])
    for up, down in zip(range(0, len(poly), +1), range(len(poly)-1, -1, -1)):
        if poly[up] == 0.0 and poly[down] == 0.0:
            to_return[up] = False
            to_return[down] = False
        else:
            break
    
    return poly[to_return]

def fit_invariant(A, B, kernel, order, Pinit, powers='+'):
    '''
    This function fits the parameters of a polynomial of the type xi = Σ [pi]_k t**k,
    of maximum order order (positive and negative coefficients), in order to solve the equation 
    (t A - B) m(x) = 0, where m(x) is the kernel function of x.
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

        num_powers_of_lambda = max_degree+1
        f = np.zeros(n*num_powers_of_lambda)
        for i in range(n):
            poly_line = Poly(0.0)
            for j, alpha in enumerate(kernel):
                mj_poly = np.prod([ x_poly[dim]**alpha[dim] for dim in range(n) ])
                poly_line += polyAB[i][j] * mj_poly
            poly_line.coef = np.hstack([ poly_line.coef, np.zeros( max_degree - poly_line.degree() ) ])
            f[num_powers_of_lambda*i:num_powers_of_lambda*(i+1)] = poly_line.coef

        return f

    polyAB_neg = [[ [ 0.0, -B[i,j], A[i,j] ] for j in range(p) ] for i in range(n) ]

    def f_neg(params):
        ''' Objective function '''
        P = params.reshape((n, 2*order+1))
    
        ''' List of polynomials representing the solution for each state '''
        x_poly = [ P[i,:] for i in range(n) ]

        num_powers_of_lambda = 2*max_degree+1
        f = np.zeros(n*num_powers_of_lambda)
        for i in range(n):
            poly_line = np.zeros(num_powers_of_lambda)
            for j, alpha in enumerate(kernel):
                
                partials = [ np.array([ 1.0 ])  for _ in range(n) ]
                for dim in range(n):
                    for _ in range(alpha[dim]):
                        partials[dim] = trim( np.convolve(x_poly[dim], partials[dim]) )

                mj_poly = np.array([ 1.0 ])
                for dim in range(n): 
                    mj_poly = trim( np.convolve(partials[dim], mj_poly) )

                poly_line_term = trim( np.convolve(polyAB_neg[i][j], mj_poly) )
                remaining_zeros = np.zeros(int(max_degree - (len(poly_line_term)-1)/2))
                poly_line += np.hstack([ remaining_zeros, poly_line_term, remaining_zeros ])

                f[num_powers_of_lambda*i:num_powers_of_lambda*(i+1)] = poly_line

        return f

    param_init = Pinit.flatten()
    
    if powers == '+': 
        sol = least_squares( f, param_init )
        print(sol)
        return sol.x.reshape((n, order+1))

    if powers == '-': 
        sol = least_squares( f_neg, param_init )
        print(sol)
        return sol.x.reshape((n, 2*order+1))

# ------------------------------------ Test invariant fitting -----------------------------------
n, d = 2, 1
kernel = Kernel(dim=n, degree=d)
print(kernel)
kernel_powers = kernel._powers
p = len(kernel_powers)

clf_eigen = [6, 1]
clf_angle = 20
clf_center = np.array([0.0, 0.0]).reshape((2,1))
Rv = rot2D( np.deg2rad(clf_angle) )
Hv = Rv.T @ np.diag(clf_eigen) @ Rv
B = Hv @ np.hstack([ -clf_center, np.eye(n) ])
clf = KernelLyapunov(kernel=kernel, P=kernel_quadratic(clf_eigen, Rv, clf_center, p), limits=limits, spacing=0.1 )

cbf_eigen = [3, 1]
cbf_angle = 0
cbf_center = np.array([0.0, 4]).reshape((2,1))
Rh = rot2D( np.deg2rad(cbf_angle) )
Hh = Rh.T @ np.diag(cbf_eigen) @ Rh
A = Hh @ np.hstack([ -cbf_center, np.eye(n) ])
cbf = KernelBarrier(kernel=kernel, Q=kernel_quadratic(cbf_eigen, Rh, cbf_center, p), limits=limits, spacing=0.1 )
cbfs = [cbf]

fx, fy = 0.0, 0.0                       # constant force with fx, fy components
F = np.zeros([p,p])
F[1,0], F[2,0] = fx, fy
def g(state):
    return np.eye(2)
plant = KernelAffineSystem(initial_state=[0.0,0.0], initial_control=[0.0, 0.0], kernel=kernel, F=F, g_method=g)
kerneltriplet = KernelFamily( plant=plant, clf=clf, cbfs=cbfs, 
                               params={"slack_gain": 1.0, "clf_gain": 1.0, "cbf_gain": 1.0}, limits=limits, spacing=0.2 )

# ------------------------------------ Plotting -----------------------------------
cbf.plot_levels(ax=ax, levels=[0.0])
kerneltriplet.plot_invariant(ax, cbf_index=0)

init_x_plot, = ax.plot([],[],'ob', alpha=0.5)
invariant_plot, = ax.plot([],[],'r.-')

while True:
    pt = plt.ginput(1, timeout=0)
    init_x = [ pt[0][0], pt[0][1] ]
    init_x_plot.set_data([init_x[0]], [init_x[1]])

    if "clf_contour" in locals():
        for coll in clf_contour:
            coll.remove()
        del clf_contour

    order = 12
    max_degree = order+1
    # max_degree = 2*order+1
    Pinit = np.random.randn(n, max_degree)
    P = fit_invariant(A, B, kernel_powers, order, Pinit, powers='+')

    x_list = [ np.sum([ P[0,i] * (t**i) for i in range(order+1-max_degree, order+1) ]) for t in t_list ]
    y_list = [ np.sum([ P[1,i] * (t**i) for i in range(order+1-max_degree, order+1) ]) for t in t_list ]
    invariant_plot.set_data(x_list, y_list)

    l = kerneltriplet.lambda_fun(init_x, cbf_index = 0)
    print(f"Lambda = {l}")

    V = clf.function(init_x)
    clf_contour = clf.plot_levels(ax=ax, levels=[V])

    plt.pause(0.001)