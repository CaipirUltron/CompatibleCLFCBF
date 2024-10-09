import numpy as np

from dynamic_systems import PolyAffineSystem
from functions import LeadingShape, Kernel, KernelQuadratic, KernelLyapunov, KernelBarrier, KernelFamily, MultiPoly
from controllers import NominalQP
from common import kernel_quadratic, rot2D, box, load_compatible, PSD_closest, NSD_closest

initial_state = [-5.0, 9.38]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

limits = 12*np.array((-1,1,-1,1))

# ---------------------------------------------- Define kernel function ----------------------------------------------------
kernel = Kernel(dim=n, degree=2)
kernel_dim = kernel._num_monomials
powers = kernel._powers
print(kernel)

# -------------------------------------------------- Define system ---------------------------------------------------------
EYE = np.eye(n)

f = MultiPoly( kernel=powers, coeffs=[ np.zeros(n) for _ in range(kernel_dim) ] )
# f = MultiPoly( kernel=powers, coeffs=[ np.zeros(n) ] + [ EYE[k,:] for k in range(n) ] + [ np.zeros(n) for _ in range(kernel_dim-(n+1)) ] )

g = MultiPoly( kernel=powers, coeffs=[ np.eye(n) ] + [ np.zeros((n,n)) for _ in range(kernel_dim-1) ] )

plant = PolyAffineSystem(initial_state=initial_state, initial_control=initial_control, f=f, g=g)

# ---------------------------------------------------- Define CLF ----------------------------------------------------------
clf_center = [0.0, -3.0]
clf_eig = np.array([ 1.0, 1.0 ])
clf_angle = np.deg2rad(-45)
Pquadratic = kernel_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=kernel_dim)
clf = KernelLyapunov(kernel=kernel, P=Pquadratic, limits=limits)
clf_poly = clf.to_multipoly()

# clf_poly = MultiPoly.load("wierd")
# clf = KernelLyapunov.from_multipoly(clf_poly, limits=limits, spacing=0.1)

# ----------------------------------------------------- Define CBFs --------------------------------------------------------
# cbf_poly = MultiPoly.load("box_shaped")
cbf_poly = MultiPoly.load("rotated_box")

cbf = KernelBarrier.from_multipoly(poly=cbf_poly, limits=limits, spacing=0.1)
cbfs = [ cbf ]

# ------------------------------------------------- Define controller ------------------------------------------------------
sample_time = .005
p = 1.0
kerneltriplet = KernelFamily( plant=plant, clf=clf, cbfs=cbfs, params={ "slack_gain": p }, limits=limits, spacing=0.4 )
controller = NominalQP(kerneltriplet, dt=sample_time)
T = 15

# ---------------------------------------------  Configure plot parameters -------------------------------------------------
plot_config = {
    "figsize": (5,5), 
    "gridspec": (1,1,1), 
    "widthratios": [1], 
    "heightratios": [1], 
    "limits": limits,
    "path_length": 10, 
    "numpoints": 1000, 
    "drawlevel": True, 
    "resolution": 50, 
    "fps":30, "pad":2.0, 
    "invariant": True, 
    "equilibria": True, 
    "arrows": False,
}

logs = { "sample_time": sample_time, "P": clf.P.tolist(), "Q": [ cbf.Q.tolist() for cbf in cbfs ] }