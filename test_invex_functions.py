import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
from functions import Kernel, KernelLyapunov, KernelQuadratic
from common import lyap, create_quadratic, rot2D, symmetric_basis

# np.set_printoptions(precision=4, suppress=True)
limits = 12*np.array((-1,1,-1,1))

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Test Invex CLFs")
# ax.set_aspect('equal', adjustable='box')

xmin, xmax = -100, 100
fmin, fmax = -100, 1000
ax.set_xlim(xmin, xmax)
ax.set_ylim(fmin, fmax)

def gen_random(num_pts):
    pts = []
    for _ in range(num_pts):
        x = np.random.randint(low=xmin, high=xmax)
        val = np.random.randint(low=0.1*fmin, high=0.1*fmax)
        pts.append( (x,val) )
    return pts

# ------------------------------------ Define kernel and CLF -----------------------------------
kernel = Kernel(dim=2, degree=2)
print(kernel)
kernel_dim = kernel._num_monomials

for k, A in enumerate(kernel.get_A_matrices()):
    print(f"A{k+1} = {A}")

eigenvalues = [0.01]
center = [0]
R = np.array([[1]])
Pinit = create_quadratic(eigenvalues, R, center, kernel_dim)

delta_fun = KernelQuadratic(kernel=kernel, coefficients=np.zeros([kernel_dim,kernel_dim]), limits=limits )
clf = KernelLyapunov(kernel=kernel, P=Pinit, limits=limits )

#--------------------------------------- Optimization probl. ------------------------------------
delta_var = cp.Variable( (kernel_dim, kernel_dim), symmetric=True )
Pcurr_var = cp.Parameter( (kernel_dim, kernel_dim), symmetric=True )
# delta_nom_var = cp.Parameter( (kernel_dim, kernel_dim), symmetric=True )

epsilon = 1e-0
delta_spam = 1e-3
# cost = cp.norm( delta_var - delta_nom_var )
cost = 0.0

num_pts = 5
m_vars, val_vars, pt_plots = [], [], []
for _ in range(num_pts):
    m_vars.append(cp.Parameter(kernel_dim))
    val_vars.append(cp.Parameter())
    cost += cp.norm( m_vars[-1].T @ delta_var @ m_vars[-1] - 2*val_vars[-1] )
    
    pt_graph, = ax.plot([], [], "k*", alpha=0.6)
    pt_plots.append(pt_graph)

constraints = [ Pinit + delta_var >> 0 ]
# constraints += [ cp.lambda_max(delta_var) <= delta_spam ]
# constraints += [ cp.lambda_min(delta_var) >= -delta_spam ]

m = kernel.function(center)
constraints += [ m.T @ delta_var @ m == 0 ]

prob = cp.Problem( cp.Minimize(cost), constraints )

#---------------------------------------  ------------------------------------
step = 0.1
xrange = np.arange(xmin, xmax, step)
frange = [ clf.function([x]) for x in xrange ]
delta_range = [ delta_fun.function([pt]) for pt in xrange ]

clf_graph, = ax.plot(xrange, frange, color = "b", alpha=0.6)
delta_graph, = ax.plot(xrange, delta_range, color = "r", alpha=0.6)
ax.legend([clf_graph, delta_graph], ["f0(x) + Δ(x)","Δ(x)"])

N = 100
for i in range(N):
    
    plt.pause(1)

    # delta_nom_var.value = np.zeros([kernel_dim, kernel_dim])
    # for basis in symmetric_basis(kernel_dim): 
    #     delta_nom_var.value += delta_spam*np.random.randn()*basis

    for k, (pt,val) in enumerate(gen_random(num_pts)):
        m_vars[k].value = kernel.function([pt])
        val_vars[k].value = val
        pt_plots[k].set_data([pt],[val])

    prob.solve(solver="SCS", verbose=True, max_iters=10000)

    Pcurr_var.value = Pinit + delta_var.value

    print(f"λ(P) = {np.linalg.eigvals(Pcurr_var.value)}")

    clf.set_params(P=Pcurr_var.value)
    delta_fun.set_params(coefficients=delta_var.value)

    print(f"λ(Δ) = {np.linalg.eigvals(delta_var.value)}")

    delta_range = [ delta_fun.function([pt]) for pt in xrange ]
    delta_graph.set_data(xrange, delta_range)

    frange = [ clf.function([pt]) for pt in xrange ]
    clf_graph.set_data(xrange, frange)

plt.show()