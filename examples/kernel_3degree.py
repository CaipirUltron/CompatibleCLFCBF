import numpy as np

from shapely import LineString, LinearRing, Polygon

from controllers import NominalQP
from dynamic_systems import KernelAffineSystem
from functions import LeadingShape, Kernel, KernelLyapunov, KernelBarrier, KernelTriplet
from common import create_quadratic, circular_boundary_shape, rot2D, polygon, load_compatible, discretize, segmentize, enclosing_circle

initial_state = [0.2, 2.5]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

# limits = np.array(((-7.5, 7.5,-4, 5))
limits = 30*np.array((-1,1,-1,1))

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=n, degree=3)
kernel_dim = kernel._num_monomials
print(kernel)

# -------------------------------------------------- Define system ---------------------------------------------------------
fx, fy = 0.0, 0.0                       # constant force with fx, fy components
F = np.zeros([kernel_dim,kernel_dim])
F[1,0], F[2,0] = fx, fy

def g(state):
    return np.eye(m)

plant = KernelAffineSystem(initial_state=initial_state, initial_control=initial_control, kernel=kernel, F=F, g_method=g)

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
clf_center = [0.0, -5.0]
base_level = 25

points = []
points.append({ "coords": [ 0.0,  1.0], "gradient": [ 0.0,  1.0], "curvature": -0.3 })
points.append({ "coords": [-5.0,  3.0], "gradient": [-1.0,  1.0] })
points.append({ "coords": [ 5.0,  3.0], "gradient": [ 1.0,  1.0] })
points.append({ "coords": [ 5.0,  0.0], "gradient": [ 1.0, -1.0] })
points.append({ "coords": [-5.0,  0.0], "gradient": [-1.0, -1.0] })

clf_eig = np.array([ 8.0, 1.0 ])
clf_angle = np.deg2rad(0)
Pquadratic = create_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim)

clf = KernelLyapunov(kernel=kernel, P=Pquadratic, limits=limits)
# clf = KernelLyapunov(kernel=kernel, points=points, centers=[clf_center], limits=limits)
# clf = KernelLyapunov(kernel=kernel, points=points, centers=[clf_center], leading=LeadingShape(Pquadratic,approximate=True), limits=limits)
clf.is_SOS_convex(verbose=True)

# ------------------------------------------- Define CBF for U-shaped obstacle ---------------------------------------------
center = (0, 0)

skeleton_line = LineString([(-4, 3), (-4, 0), center, (4, 0), (4, 3)])
skeleton_pts = discretize(skeleton_line, spacing=0.4)
skeleton_segs = segmentize(skeleton_pts, center)

obstacle_poly = skeleton_line.buffer(1.0, cap_style='flat')
boundary_pts = discretize(obstacle_poly, spacing=0.4)

shape_matrix = circular_boundary_shape( radius=6, center=center, kernel_dim=kernel_dim )
leading = LeadingShape(shape_matrix, bound='lower')

quadratic_cbf = KernelBarrier(kernel=kernel, Q=shape_matrix, limits=limits, spacing=0.1)

cbf = KernelBarrier(kernel=kernel, boundary=boundary_pts, skeleton=skeleton_segs, 
                    leading=leading, limits=limits, spacing=0.1)

cbf.is_bounded_by(leading.shape,verbose=True)
cbf.is_SOS_convex(verbose=True)

# ------------------------------------------------- Define controller ------------------------------------------------------
sample_time = .002
p, alpha, beta = 1.0, 1.0, 1.0
kerneltriplet = KernelTriplet( plant=plant, clf=clf, cbf=cbf,
                              params={"slack_gain": p, "clf_gain": alpha, "cbf_gain": beta},
                              limits=limits.tolist(), spacing=0.1)

controller = NominalQP(kerneltriplet, dt=sample_time)
T = 15

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
plot_config = {
    "figsize": (5,5), "gridspec": (1,1,1), "widthratios": [1], "heightratios": [1], "limits": limits,
    "path_length": 10, "numpoints": 1000, "drawlevel": True, "resolution": 50, "fps":30, "pad":2.0, "invariants": True, "equilibria": True, "arrows": True
}
logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": cbf.Q.tolist() }