import numpy as np

from dynamic_systems import KernelAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier, KernelFamily
from controllers import NominalQP
from common import create_quadratic, rot2D, load_compatible

initial_state = [0.2, 6.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

limits = 12*np.array((-1,1,-1,1))

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=n, degree=1)
kernel_dim = kernel._num_monomials
print(kernel)

# -------------------------------------------------- Define system ---------------------------------------------------------
fx, fy = 0.0, 0.0                       # constant force with fx, fy components
F = np.zeros([kernel_dim,kernel_dim])
F[1,0], F[2,0] = fx, fy

def g(state):
    return np.eye(m)
plant = KernelAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

# --------------------------------------------- Define CLF (quadratic) -----------------------------------------------------
clf_eig = np.array([ 3.0, 1.0 ])
clf_angle = 0
clf_center = [0.0, -2.0]
Pquadratic = create_quadratic( eigen=clf_eig, R=rot2D(np.deg2rad(clf_angle)), center=clf_center, kernel_dim=kernel_dim )

clf = KernelLyapunov(kernel=kernel, P=load_compatible( __file__, Pquadratic, load_compatible = True ), limits=limits )
# clf = KernelLyapunov(kernel=kernel, P=Pquadratic, limits=limits)

# --------------------------------------------- Define CBF (quadratic) -----------------------------------------------------
cbf_eig = 0.2*np.array([ 0.2, 1.2 ])
cbf_center = [0.0, 3.0]
cbf_angle = np.deg2rad(5)

Qquadratic = create_quadratic(eigen=cbf_eig, R=rot2D(cbf_angle), center=cbf_center, kernel_dim=kernel_dim)
cbf = KernelBarrier(kernel=kernel, Q=Qquadratic, limits=limits)

cbfs = [ cbf ]
# -------------------------------------------Define triplet and controller -------------------------------------------------
sample_time = .002
p, alpha, beta = 1.0, 1.0, 1.0
kerneltriplet = KernelFamily( plant=plant, clf=clf, cbfs=cbfs, 
                               params={"slack_gain": p, "clf_gain": alpha, "cbf_gain": beta}, 
                               limits=limits, spacing=0.2 )

controller = NominalQP(kerneltriplet, dt=sample_time)
T = 15

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
plot_config = {
    "figsize": (5,5), "gridspec": (1,1,1), "widthratios": [1], "heightratios": [1], "limits": limits,
    "path_length": 10, "numpoints": 1000, "drawlevel": True, "resolution": 50, "fps":30, "pad":2.0, "invariants": True, "equilibria": True, "arrows": True
}

logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": [ cbf.Q.tolist() for cbf in cbfs ] }