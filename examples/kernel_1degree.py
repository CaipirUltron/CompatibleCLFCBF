import numpy as np

from dynamic_systems import PolyAffineSystem
from functions import Kernel, KernelLyapunov, KernelBarrier, KernelFamily, MultiPoly
from controllers import NominalQP
from common import kernel_quadratic, rot2D, load_compatible

initial_state = [-5.0, 6.0]
initial_control = [0.0, 0.0]
n = len(initial_state)
m = len(initial_control)

limits = 12*np.array((-1,1,-1,1))

# ----------------------------------------- Define baseline kernel function ------------------------------------------------
kernel = Kernel(dim=n, degree=1)
kernel_dim = kernel._num_monomials
powers = kernel._powers
print(kernel)

# -------------------------------------------------- Define system ---------------------------------------------------------
EYE = np.eye(n)

f = MultiPoly( kernel=powers, coeffs=[ np.zeros(n) for _ in range(kernel_dim) ] )
# f = MultiPoly( kernel=powers, coeffs=[ np.zeros(n) ] + [ EYE[k,:] for k in range(n) ] + [ np.zeros(n) for _ in range(kernel_dim-(n+1)) ] )

g = MultiPoly( kernel=powers, coeffs=[ np.eye(n) ] + [ np.zeros((n,n)) for _ in range(kernel_dim-1) ] )

plant = PolyAffineSystem(initial_state=initial_state, initial_control=initial_control, f=f, g=g)

# --------------------------------------------- Define CLF (quadratic) -----------------------------------------------------
clf_eig = np.array([ 4.0, 1.0 ])
clf_angle = 0
clf_center = [0.0, -2.0]
Pquadratic = kernel_quadratic( eigen=clf_eig, R=rot2D(np.deg2rad(clf_angle)), center=clf_center, kernel_dim=kernel_dim )

clf = KernelLyapunov(kernel=kernel, P=Pquadratic, limits=limits)
# clf = KernelLyapunov(kernel=kernel, P=load_compatible( __file__, Pquadratic, load_compatible = True ), limits=limits )

# --------------------------------------------- Define CBF (quadratic) -----------------------------------------------------
cbf_eig = 1*np.array([ 0.2, 1.2 ])
cbf_center = [0.0, 3.0]
cbf_angle = np.deg2rad(30)

Qquadratic = kernel_quadratic(eigen=cbf_eig, R=rot2D(cbf_angle), center=cbf_center, kernel_dim=kernel_dim)
cbf = KernelBarrier(kernel=kernel, Q=Qquadratic, limits=limits)

cbfs = [ cbf ]
# ----------------------------------------- Define triplet and controller --------------------------------------------------
sample_time = .005
p = 1.0
kerneltriplet = KernelFamily( plant=plant, clf=clf, cbfs=cbfs, params={ "slack_gain": p }, limits=limits, spacing=0.2 )
controller = NominalQP(kerneltriplet, dt=sample_time)
T = 12

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