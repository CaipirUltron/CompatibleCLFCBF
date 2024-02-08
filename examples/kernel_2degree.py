import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP
from common import create_quadratic, rot2D, box

initial_state = [0.5, 6.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

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
plant = ConservativeAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
clf_center = [0.0, -3.0]
base_level = 25

points = []
points.append({ "coords": [-4.0,  6.0], "gradient": [-1.0,  0.5] })
points.append({ "coords": [ 4.0,  6.0], "gradient": [ 0.0,  6.5] })
points.append({ "coords": [ 0.0,  5.0], "gradient": [ 2.0,  6.0] })
# points.append({ "coords": [ 0.0,  -8.0], "gradient": [ 0.0,  -1.0] })
# clf = KernelLyapunov(*initial_state, kernel=kernel, points=points, centers=[clf_center])

clf_eig = 0.01*np.array([ 6.0, 1.0 ])
clf_angle = np.deg2rad(-45)
Pquadratic = create_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim)

# clf = KernelLyapunov(*initial_state, kernel=kernel, P=Pquadratic)
clf = KernelLyapunov(*initial_state, kernel=kernel, leading={ "shape": Pquadratic, "uses": ["lower_bound", "approximation"] })
clf.is_sos_convex(verbose=True)

# ----------------------------------------------------- Define CBF ---------------------------------------------------------
# Fits CBF to a box-shaped obstacle
center = [ 0.0, 0.0 ]
pts = box( center=center, height=5, width=5, angle=0, spacing=0.4, gradients=1, at_edge=True )
cbf = KernelBarrier(*initial_state, kernel=kernel, boundary=pts, centers=[center])
cbf.is_sos_convex(verbose=True)
# ------------------------------------------------- Define controller ------------------------------------------------------
T = 15
sample_time = .002
p, alpha, beta = 1.0, 1.0, 1.0
controller = NominalQP(plant, clf, cbf, alpha, beta, p, dt=sample_time)

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
limits = 15*np.array([[-1, 1],[-1, 1]])

plot_config = {
    "figsize": (5,5), "gridspec": (1,1,1), "widthratios": [1], "heightratios": [1], "limits": limits.tolist(),
    "path_length": 10, "numpoints": 1000, "drawlevel": True, "resolution": 50, "fps":30, "pad":2.0, "invariants": True, "equilibria": True, "arrows": True
}

logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": cbf.Q.tolist() }