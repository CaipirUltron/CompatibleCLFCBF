import numpy as np

from dynamic_systems import KernelAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier, KernelTriplet
from controllers import NominalQP
from common import create_quadratic, rot2D, box, load_compatible

initial_state = [0.5, 6.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)
limits = 12*np.array([[-1, 1],[-1, 1]])

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(*initial_state, degree=2)
kernel_dim = kernel.kernel_dim
print(kernel)

# -------------------------------------------------- Define system ---------------------------------------------------------
fx, fy = 0.0, 0.0                       # constant force with fx, fy components
F = np.zeros([kernel_dim,kernel_dim])
F[1,0], F[2,0] = fx, fy

def g(state):
    return np.eye(m)
plant = KernelAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
clf_center = [0.0, -3.0]
base_level = 25

# points = []
# points.append({ "coords": [-4.0,  6.0], "gradient": [-1.0,  0.5] })
# points.append({ "coords": [ 4.0,  6.0], "gradient": [ 0.0,  6.5] })
# points.append({ "coords": [ 0.0,  5.0], "gradient": [ 2.0,  6.0] })
# points.append({ "coords": [ 0.0,  -8.0], "gradient": [ 0.0,  -1.0] })
# clf = KernelLyapunov(*initial_state, kernel=kernel, points=points, centers=[clf_center])

clf_eig = np.array([ 1.0, 1.0 ])
clf_angle = np.deg2rad(-45)
Pquadratic = create_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim)

clf = KernelLyapunov(*initial_state, kernel=kernel, P=load_compatible(__file__, Pquadratic, load_compatible=True))
# clf = KernelLyapunov(*initial_state, kernel=kernel, leading={ "shape": Pquadratic, "uses": ["lower_bound", "approximation"] })
clf.is_sos_convex(verbose=True)

# ----------------------------------------------------- Define CBF ---------------------------------------------------------
# Fits CBF to a box-shaped obstacle
center = [ 0.0, 2.0 ]
pts = box( center=center, height=5, width=5, angle=10, spacing=0.4, gradients=1, at_edge=True )
cbf = KernelBarrier(*initial_state, kernel=kernel, boundary=pts, centers=[center])
cbf.is_sos_convex(verbose=True)

# ------------------------------------------------- Define controller ------------------------------------------------------
sample_time = .002
p, alpha, beta = 1.0, 1.0, 1.0
kerneltriplet = KernelTriplet( plant=plant, clf=clf, cbf=cbf,
                              params={"slack_gain": p, "clf_gain": alpha, "cbf_gain": beta},
                              limits=limits.tolist(), spacing=0.2 )

controller = NominalQP(kerneltriplet, dt=sample_time)
T = 15

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
plot_config = {
    "figsize": (5,5), "gridspec": (1,1,1), "widthratios": [1], "heightratios": [1], "limits": limits.tolist(),
    "path_length": 10, "numpoints": 1000, "drawlevel": True, "resolution": 50, "fps":30, "pad":2.0, "invariants": True, "equilibria": True, "arrows": True
}

logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": cbf.Q.tolist() }