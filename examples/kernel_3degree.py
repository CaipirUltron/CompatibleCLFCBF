import numpy as np

from dynamic_systems import KernelAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier, KernelTriplet
from controllers import NominalQP
from common import create_quadratic, rot2D, polygon, load_compatible

initial_state = [0.2, 2.5]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)
# limits = np.array([[-7.5, 7.5],[-4, 5]])
limits = 20*np.array([[-1, 1],[-1, 1]])

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(*initial_state, degree=3)
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
clf_center = [0.0, -5.0]
base_level = 25

points = []
points.append({ "coords": [ 0.0,  1.0], "gradient": [ 0.0,  1.0], "curvature": -0.3 })
points.append({ "coords": [-5.0,  3.0], "gradient": [-1.0,  1.0] })
points.append({ "coords": [ 5.0,  3.0], "gradient": [ 1.0,  1.0] })
points.append({ "coords": [ 5.0,  0.0], "gradient": [ 1.0,  -1.0] })
points.append({ "coords": [-5.0,  0.0], "gradient": [-1.0,  -1.0] })

clf_eig = np.array([ 8.0, 1.0 ])
clf_angle = np.deg2rad(0)
Pquadratic = create_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim)

clf = KernelLyapunov(*initial_state, kernel=kernel, P=load_compatible(__file__, Pquadratic, load_compatible=True))
# clf = KernelLyapunov(*initial_state, kernel=kernel, points=points, centers=[clf_center])
# clf = KernelLyapunov(*initial_state, kernel=kernel, points=points, centers=[clf_center], leading={ "shape": Pquadratic, "uses": ["approximation"] })
clf.is_SOS_convex(verbose=True)

# ----------------------------------------------------- Define CBF ---------------------------------------------------------
# Fits CBF to a U shaped obstacle

# centers = [ [0.0, 0.0] ]
centers = [ [-3.0, 0.0 ],[ 3.0, 0.0 ], [ -4.0, 3.0 ], [ 4.0, 3.0 ] ]
vertices = [ [ 5.0,-1.0 ], [ 5.0, 3.0 ], [ 3.0, 3.0 ], [ 3.0, 1.0 ],
             [-3.0, 1.0 ], [-3.0, 3.0 ], [-5.0, 3.0 ], [-5.0,-1.0 ] ]
pts = polygon( vertices=vertices, spacing=0.2, closed=True, gradients=0, at_edge=False )

def line(start, end, sep):
    num_pts = round( np.linalg.norm( np.array(start) - np.array(end)) / sep )
    x_coords = [ x for x in np.linspace(start[0], end[0], num_pts).tolist() ]
    y_coords = [ y for y in np.linspace(start[1], end[1], num_pts).tolist() ]
    return list(zip(x_coords,y_coords))

pt_sep = 0.2
skeleton = []
skeleton.append( line((0,0), (4,0), pt_sep ) )
skeleton.append( line((4,0), (5,-1), pt_sep ))
skeleton.append( line((4,0), (4,2), pt_sep ))
skeleton.append( line((4,2), (5,3), pt_sep ))
skeleton.append( line((4,2), (3,3), pt_sep ))

skeleton.append( line((0,0), (-4,0), pt_sep ))
skeleton.append( line((-4,0), (-5,-1), pt_sep ))
skeleton.append( line((-4,0), (-4,2), pt_sep ))
skeleton.append( line((-4,2), (-5,3), pt_sep ))
skeleton.append( line((-4,2), (-3,3), pt_sep ))

pt_sep = 1.0
min_x, max_x = limits[0,0], limits[0,1]
min_y, max_y = limits[1,0], limits[1,1]

safe_points = []
safe_points += line((max_x, max_y), (max_x, min_y), pt_sep )
safe_points += line((max_x, min_y), (min_x, min_y), pt_sep )
safe_points += line((min_x, min_y), (min_x, max_y), pt_sep )
safe_points += line((min_x, max_y), (max_x, max_y), pt_sep )

cbf_center = [0.0, 1.0]
# cbf_eig = 0.03*np.array([ 1.0, 1.0 ])
cbf_eig = 0.04*np.array([ 1.0, 1.0 ])

cbf_angle = np.deg2rad(0)
Qquadratic = create_quadratic(eigen=cbf_eig, R=rot2D(cbf_angle), center=cbf_center, kernel_dim=kernel_dim)
quadratic_cbf = KernelBarrier(*initial_state, kernel=kernel, Q=Qquadratic)
cbf = KernelBarrier(*initial_state, kernel=kernel, boundary=pts, skeleton=skeleton, leading={ "shape": Qquadratic, "uses": ["lowerbound"] })
cbf.is_SOS_convex(verbose=True)

# ------------------------------------------------- Define controller ------------------------------------------------------
sample_time = .01
p, alpha, beta = 1.0, 1.0, 1.0
kerneltriplet = KernelTriplet( plant=plant, clf=clf, cbf=cbf,
                              params={"slack_gain": p, "clf_gain": alpha, "cbf_gain": beta},
                              limits=limits.tolist(), spacing=0.1)

controller = NominalQP(kerneltriplet, dt=sample_time)
T = 15

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
plot_config = {
    "figsize": (5,5), "gridspec": (1,1,1), "widthratios": [1], "heightratios": [1], "limits": limits.tolist(),
    "path_length": 10, "numpoints": 1000, "drawlevel": True, "resolution": 50, "fps":30, "pad":2.0, "invariants": True, "equilibria": True, "arrows": True
}
logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": cbf.Q.tolist() }