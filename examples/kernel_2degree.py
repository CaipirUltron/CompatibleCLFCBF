import numpy as np

from dynamic_systems import KernelAffineSystem
from functions import LeadingShape, Kernel, KernelQuadratic, KernelLyapunov, KernelBarrier, KernelFamily
from controllers import NominalQP
from common import kernel_quadratic, rot2D, box, load_compatible, NGN_decomposition, PSD_closest, NSD_closest

initial_state = [-5.0, 9.38]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

limits = 12*np.array((-1,1,-1,1))

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=n, degree=2)
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
clf_center = [0.0, -3.0]
base_level = 15

# points = []
# points.append({ "coords": [-4.0,  6.0], "gradient": [-1.0,  0.5] })
# points.append({ "coords": [ 4.0,  6.0], "gradient": [ 0.0,  6.5] })
# points.append({ "coords": [ 0.0,  5.0], "gradient": [ 2.0,  6.0] })
# points.append({ "coords": [ 0.0,  -8.0], "gradient": [ 0.0,  -1.0] })
# clf = KernelLyapunov(*initial_state, kernel=kernel, points=points, centers=[clf_center])

clf_eig = np.array([ 1.0, 1.0 ])
clf_angle = np.deg2rad(-45)
Pquadratic = kernel_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim)

# fun = KernelQuadratic(kernel=kernel, P=Pquadratic, limits=limits, spacing=0.1)
# print(fun.matrix_coefs)

# clf = KernelLyapunov(kernel=kernel, P=load_compatible(__file__, Pquadratic, load_compatible=True), limits=limits)
clf = KernelLyapunov(kernel=kernel, P=Pquadratic, limits=limits)
# clf = KernelLyapunov(kernel=kernel, leading=LeadingShape(Pquadratic,bound='lower',approximate=True), limits=limits)
clf.is_SOS_convex(verbose=True)

# ----------------------------------------------------- Define CBFs --------------------------------------------------------
# Fits CBF to a box-shaped obstacle

box_center = [ 0.0, 2.0 ]
box_angle = 30
box_height, box_width = 5, 5

init_eig = [ 0.2/box_height, 0.2/box_width ]
init_R = rot2D(np.deg2rad(box_angle))
Qinit = kernel_quadratic(eigen=init_eig, R=init_R, center=box_center, kernel_dim=kernel_dim)

print(f"Eigs of Qinit = {np.linalg.eigvals(Qinit)}")
boundary_pts = box( center=box_center, height=box_height, width=box_width, angle=box_angle, spacing=0.4 )

# cbf = KernelBarrier(kernel=kernel, Q=Qinit, limits=limits, spacing=0.1)
cbf = KernelBarrier(kernel=kernel, boundary=boundary_pts, centers=[box_center], initial_shape = Qinit, limits=limits, spacing=0.1)
# cbf = KernelBarrier(kernel=kernel, boundary=boundary_pts, skeleton=skeleton, leading=LeadingShape(Qquadratic,bound='upper'), limits=limits, spacing=0.1)

eigQ = np.linalg.eigvals(cbf.Q)
print(f"Spectra of Q = {eigQ}")

N, G, error = NGN_decomposition(n, cbf.Q)
print(f"Decomposition error norm = {np.linalg.norm(error)}")

D = kernel.D(N)
eigD = np.linalg.eigvals(D)
print(f"Spectra of D(N) = {eigD}")

cbf.set_params(Q = N.T @ G @ N)
cbf.generate_contour()

cbfs = [ cbf ]
# ------------------------------------------------- Define controller ------------------------------------------------------
sample_time = .005
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