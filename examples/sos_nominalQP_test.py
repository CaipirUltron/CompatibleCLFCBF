import sys
import json
import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP

initial_state = [4.2, 5.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# ---------------------------------------------- Define kernel function ----------------------------------------------------
max_degree = 2
kernel = Kernel(*initial_state, degree = max_degree)
p = kernel.kernel_dim
print(kernel)

# -------------------------------------------------- Define system ---------------------------------------------------------
F = np.zeros([p,p])
def g(state):
    return np.eye(m)
plant = ConservativeAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

gen_random = False
if gen_random:
    Proot = np.random.rand(p,p)
    P = Proot.T @ Proot
    clf_center = np.array([-6.0, -6.0])

    Qroot = 0.1*np.random.rand(p,p)
    Q = 0.1*Qroot.T @ Qroot
    cbf_center = np.array([-2.0, -2.0])
else:
    simulation_file = sys.argv[1].replace(".json","")
    location = simulation_file+".json"
    try:
        with open(location,'r') as file:
            print("here")
            print("Loading simulation with "+simulation_file + ".json")
            logs = json.load(file)
            P, Q = np.array( logs["P"] ), np.array( logs["Q"] )
            clf_center, cbf_center = np.array( logs["clf_center"] ), np.array( logs["cbf_center"] )
    except IOError:
        print("Couldn't locate " + simulation_file + ".json")

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
clf = KernelLyapunov(*initial_state, kernel=kernel, P=P)
print( clf.define_zeros( clf_center ) )

# ---------------------------------------------------- Define CBF ----------------------------------------------------------
cbf = KernelBarrier(*initial_state, kernel=kernel, Q=Q)
print( cbf.define_zeros( cbf_center ) )

cbfs = [cbf]

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