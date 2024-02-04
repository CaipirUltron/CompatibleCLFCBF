import numpy as np

from dynamic_systems import ConservativeAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier
from controllers import NominalQP
from common import create_quadratic, rot2D

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
base_level = 16.0
points = []
points += [{ "coords": [ 0.0,  -2.0], "level": 0.0 }]
# points += [{ "coords": [ 3.0,  3.0], "level": base_level, "gradient": [ 1.0,  1.0] }]
# points += [{ "coords": [-3.0,  3.0], "level": base_level, "gradient": [-1.0,  1.0] }]
points += [{ "coords": [ 0.0,  1.0],                      "gradient": [ 0.0,  1.0], "curvature": 0.0 }]
# points += [{ "coords": [ 0.0,  5.0],                      "gradient": [ 1.8,  1.0] }]
# clf = KernelLyapunov(*initial_state, kernel=kernel, points=points)

clf_eig = 0.01*np.array([ 1.0, 4.0 ])
clf_angle = -np.pi/4
clf_center = [0.0, -5.0]
clf = KernelLyapunov(*initial_state, kernel=kernel, P=create_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim))

# ----------------------------------------------------- Define CBF ---------------------------------------------------------
# boundary_points = [ [-4.0, 0.0], [-4.0, -1.0], [-2.0, 0.5], [2.0, 0.5], [4.0, -1.0], [4.0, 0.0], [0.0, 1.0], [0.0, -0.5] ]   # (sad smile)
# boundary_points = [ [-4.0, 0.0], [-4.0, 1.0], [-2.0, -0.5], [2.0, -0.5], [4.0, 1.0], [4.0, 0.0] ]                            # (bell shaped)

c = np.array([3.0, 0.0])
height, width = 4, 4
t = (height/2)*np.array([ 0, +1 ])
b = (height/2)*np.array([ 0, -1 ])
l = (width/2)*np.array([-1,  0 ])
r = (width/2)*np.array([+1,  0 ])
tl = t+l
tr = t+r
bl = b+l
br = b+r

points = []
points.append( {"coords": t+c, "gradient": t} )
points.append( {"coords": b+c, "gradient": b} )
points.append( {"coords": l+c, "gradient": l} )
points.append( {"coords": r+c, "gradient": r} )
points.append( {"coords": tl+c, "gradient": tl} )
points.append( {"coords": tr+c, "gradient": tr} )
points.append( {"coords": bl+c, "gradient": bl} )
points.append( {"coords": br+c, "gradient": br} )

cbf = KernelBarrier(*initial_state, kernel=kernel, boundary=points)

# ------------------------------------------------- Define controller ------------------------------------------------------
T = 15
sample_time = .002
p, alpha, beta = 1.0, 1.0, 1.0
controller = NominalQP(plant, clf, cbf, alpha, beta, p, dt=sample_time)

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
xlimits, ylimits = [-6, 6], [-4, 8]
plot_config = {
    "figsize": (5,5), "gridspec": (1,1,1), "widthratios": [1], "heightratios": [1], "limits": [xlimits, ylimits],
    "path_length": 10, "numpoints": 1000, "drawlevel": True, "resolution": 50, "fps":30, "pad":2.0, "invariants": True, "equilibria": True, "arrows": True
}

logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": cbf.Q.tolist() }