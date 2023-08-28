import time
import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP
from common import rot2D

initial_state = [4.2, 5.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
max_degree = 2
kernel = Kernel(*initial_state, degree = max_degree)
p = kernel.kernel_dim
print(kernel)
print(kernel.alpha)

# -------------------------------------------------- Define system ---------------------------------------------------------
F = np.zeros([p,p])
def g(state):
    return np.eye(m)
plant = ConservativeAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

clf_eig = [1.0, 5.0]
clf_center = np.array([-6.0, 0.0])
P = np.zeros([p,p])
P[0,0] = clf_center.T @ np.diag(clf_eig) @ clf_center
for k in range(n):
    P[0,k+1] = -clf_center[k]*clf_eig[k]
    P[k+1,0] = -clf_center[k]*clf_eig[k]
    P[k+1,k+1] = clf_eig[k]

cbf_eig = [4.0, 1.0]
cbf_center = np.array([0.0, 0.0])
Q = np.zeros([p,p])
Q[0,0] = cbf_center.T @ np.diag(cbf_eig) @ cbf_center
for k in range(n):
    Q[0,k+1] = -cbf_center[k]*cbf_eig[k]
    Q[k+1,0] = -cbf_center[k]*cbf_eig[k]
    Q[k+1,k+1] = cbf_eig[k]

# P = np.zeros([p,p])
# angle = np.random.rand()
# P[0,0], P[1:n+1,1:n+1] = 0, rot2D(angle).T @ np.diag([10.0, 1.0]) @ rot2D(angle)

# Q = np.zeros([p,p])
# Q[0,0], Q[1:n+1,1:n+1] = 0, 0.1*np.diag([1.0, 50.0])
# cbf_center = np.array([-2.0, -2.0])

# gen_random = True
# if gen_random:
#     Proot = np.random.rand(p,p)
#     P = Proot.T @ Proot
#     clf_center = np.array([-6.0, -6.0])

#     Qroot = 0.1*np.random.rand(p,p)
#     Q = 0.1*Qroot.T @ Qroot
#     cbf_center = np.array([-2.0, -2.0])
# else:
#     simulation_file = sys.argv[1].replace(".json","")
#     location = "logs/"+simulation_file+".json"
#     try:
#         with open(location,'r') as file:
#             print("here")
#             print("Loading simulation with "+simulation_file + ".json")
#             logs = json.load(file)
#             P, Q = np.array( logs["P"] ), np.array( logs["Q"] )
#             clf_center, cbf_center = np.array( logs["clf_center"] ), np.array( logs["cbf_center"] )
#     except IOError:
#         print("Couldn't locate " + simulation_file + ".json")

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
clf = KernelLyapunov(*initial_state, kernel=kernel, P=P)
# print( clf.define_zeros( clf_center ) )

# ---------------------------------------------------- Define CBF ----------------------------------------------------------
cbf = KernelBarrier(*initial_state, kernel=kernel, Q=Q)
# print( cbf.define_zeros( cbf_center ) )

cbfs = [cbf]
from controllers import compute_equilibria_algorithm5
sol = compute_equilibria_algorithm5( plant, clf, cbf, [0.0, 5.6] )
print(sol)

z = np.array(sol["boundary"]["z"])

# ------------------------------------------------- Define controller ------------------------------------------------------
sample_time = .001
controller = NominalQP(plant, clf, cbfs, alpha = 10.0, beta = 10.0, p = 1.0)

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
xlimits, ylimits = [-8, 8], [-8, 8]
plot_config = {
    "figsize": (5,5),
    "gridspec": (1,1,1),
    "widthratios": [1],
    "heightratios": [1],
    "axeslim": tuple(xlimits+ylimits),
    "path_length": 10,
    "numpoints": 1000,
    "drawlevel": True,
    "resolution": 50,
    "fps":120,
    "pad":2.0,
    "equilibria": False
}

logs = { "sample_time": sample_time, "P": P.tolist(), "Q": Q.tolist(), "clf_center": clf_center.tolist(), "cbf_center": cbf_center.tolist() }