import scipy as sp
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
t_list = np.linspace(0, 5, N)

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

def kernel_fun(x, kernel):
    ''' Computes kernel function value '''
    p = len(kernel)
    m = np.zeros(p)
    for k, alpha in enumerate(kernel):
        m[k] = np.prod([ x[dim]**alpha[dim] for dim in range(n) ])
    return m

def fit_nullspace(kernel, M, x_init = None):

    n = M.shape[0]
    if x_init == None: x_init = np.random.randn(n)
    NullM = sp.linalg.null_space(M)
    beta_init = np.random.randn( NullM.shape[1] )

    def nullspace_cost(var, M):
        x = var[0:n]
        beta = var[n:].reshape(-1,1)
        if len(beta) != NullM.shape[1]:
            raise Exception("Matrix sizes are incorrect.")
        return np.linalg.norm( M @ kernel_fun(x, kernel) - NullM @ beta )
    
    var_init = np.hstack([x_init, beta_init])
    sol = least_squares( lambda var: nullspace_cost(var, M), var_init, method='dogbox' )
    sol_x = sol.x[0:n]
    sol_error = np.linalg.norm( M @ kernel_fun(sol_x, kernel) )

    return sol_x, sol_error    

def fit_invariant(A, B, kernel, order, init_params, powers='+'):
    '''
    This function fits the parameters of a polynomial of the type xi = Σ [pi]_k t**k,
    of maximum order order (positive and negative coefficients), in order to solve the equation 
    (t A - B) m(x) = 0, where m(x) is the kernel function of x.
    '''
    if A.shape != B.shape:
        raise Exception("Matrix sizes are not compatible.")
    n, p = A.shape[0], A.shape[1]
    
    if len(kernel) != p:
        raise Exception("Kernel and matrix sizes are not compatible.")
    
    '''
    P is the parameter matrix to be fitted.
    Each line represents the parameter for each state xi.
    Each column represents the corresponding power of t, from 0 to kappa.
    '''
    kernel_degrees = [ sum(alpha) for alpha in kernel ]
    max_degree = order*max(kernel_degrees) + 1

    x_null_A, errorA = fit_nullspace(kernel, A)
    x_null_B, errorB = fit_nullspace(kernel, B)

    print(f"xA = {x_null_A}, with cost {errorA}")
    print(f"xB = {x_null_B}, with cost {errorB}")

    def param2P(params):
        ''' Converts from parameter vector to P matrix '''

        if len(params) != n*order:
            raise Exception("Incorrect length of parameter vector.")

        P = np.zeros((n, order+1))
        P[:,0] = x_null_B
        for k in range(1, order+1):
            P[:,k] = params[ (k-1)*n : k*n ]

        return P

    polyAB = np.array([[ Poly([ -B[i,j], A[i,j] ]) for j in range(p) ] for i in range(n) ])
    def f_pos(params):
        ''' Objective function '''

        '''
        Create P matrix from parameters and a list of polynomials representing the 
        solution for each state.
        '''
        P = param2P(params)
        x_poly = [ Poly(P[i,:]) for i in range(n) ]

        '''
        Effectively computes the f function for least squares, composed of the 
        resulting lambda coefficients (must all be zero).
        '''
        num_powers_of_lambda = max_degree+1
        f = np.zeros(n*num_powers_of_lambda)
        for i in range(n):
            poly_line = Poly(0.0)
            for j, alpha in enumerate(kernel):
                mj_poly = np.prod([ x_poly[dim]**alpha[dim] for dim in range(n) ])
                poly_line += polyAB[i,j] * mj_poly
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
    
    if powers == '+': 
        sol = least_squares( f_pos, init_params, method='dogbox' )
        print(sol)
        return param2P(sol.x)

    if powers == '-': 
        sol = least_squares( f_neg, init_params, method='dogbox' )
        print(sol)
        return sol.x.reshape((n, 2*order+1))

def fit_invariant_lstsqr(A, B, kernel, order, init_params, lambda_samples):

    if A.shape != B.shape:
        raise Exception("Matrix sizes are not compatible.")
    n, p = A.shape[0], A.shape[1]
    
    if len(kernel) != p:
        raise Exception("Kernel and matrix sizes are not compatible.")
    
    x_null_B, errorB = fit_nullspace(kernel, B)
    def param2P(params):
        ''' Converts from parameter vector to P matrix '''

        if len(params) != n*(order):
            raise Exception("Incorrect length of parameter vector.")
        P = np.zeros((n, order+1))
        P[:,0] = x_null_B
        for k in range(1, order+1):
            P[:,k] = params[ (k-1)*n : k*n ]
        return P

    kernel_degrees = [ sum(alpha) for alpha in kernel ]
    max_degree = order*max(kernel_degrees) + 1

    def cost(param):

        P = param2P(param)

        f = []
        for l_sample in lambda_samples:
            lambda_vec = np.array([ l_sample**i for i in range(max_degree + 1)])
            f.append( ( l_sample * A - B ) @  kernel_fun(P @ lambda_vec, kernel) )


# ------------------------------------ Test invariant fitting -----------------------------------
n, d = 2, 1
kernel = Kernel(dim=n, degree=d)
print(kernel)
kernel_powers = kernel._powers
p = len(kernel_powers)

clf_eigen = [ 1, 5 ]
clf_angle = 45
clf_center = np.array([0.0, 0.0]).reshape((2,1))
Rv = rot2D( np.deg2rad(clf_angle) )
Hv = Rv.T @ np.diag(clf_eigen) @ Rv
B = 1.0 * Hv @ np.hstack([ -clf_center, np.eye(n) ])
clf = KernelLyapunov(kernel=kernel, P=kernel_quadratic(clf_eigen, Rv, clf_center, p), limits=limits, spacing=0.1 )

cbf_eigen = [ 1, 1 ]
cbf_angle = 5
cbf_center = np.array([0.0, 4]).reshape((2,1))
Rh = rot2D( np.deg2rad(cbf_angle) )
Hh = Rh.T @ np.diag(cbf_eigen) @ Rh
A = Hh @ np.hstack([ -cbf_center, np.eye(n) ])
cbf = KernelBarrier(kernel=kernel, Q=kernel_quadratic(cbf_eigen, Rh, cbf_center, p), limits=limits, spacing=0.1 )
cbfs = [cbf]

print(f"A = {A}\nB = {B}")

fx, fy = 0.0, 0.0                       # constant force with fx, fy components
F = np.zeros([p,p])
F[1,0], F[2,0] = fx, fy
def g(state):
    return np.eye(2)
plant = KernelAffineSystem(initial_state=[0.0,0.0], initial_control=[0.0, 0.0], kernel=kernel, F=F, g_method=g)
kerneltriplet = KernelFamily( plant=plant, clf=clf, cbfs=cbfs, params={"slack_gain": 1.0, "clf_gain": 1.0, "cbf_gain": 1.0}, limits=limits, spacing=0.2 )

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

    # mode = '+'
    # max_degree = order+1
    # init_params = np.random.randn( n*order )

    mode = '-'
    max_degree = 2*order+1
    init_params = np.random.randn( n*max_degree )

    P = fit_invariant(A, B, kernel_powers, order, init_params, powers=mode)

    x_list = [ np.sum([ P[0,i] * (t**i) for i in range(order+1-max_degree, order+1) ]) for t in t_list ]
    y_list = [ np.sum([ P[1,i] * (t**i) for i in range(order+1-max_degree, order+1) ]) for t in t_list ]
    invariant_plot.set_data(x_list, y_list)

    l = kerneltriplet.lambda_fun(init_x, cbf_index = 0)
    print(f"Lambda = {l}")

    V = clf.function(init_x)
    clf_contour = clf.plot_levels(ax=ax, levels=[V])

    plt.pause(0.001)