import numpy as np

from controllers import CompatibleQP
from dynamic_systems import LinearSystem, DriftLess
from functions import QuadraticLyapunov, QuadraticBarrier

limits = 12*np.array((-1,1,-1,1))

''' --------------------------------- Define LTI system ------------------------------------- '''
n, m = 2, 2

x0 = np.array([2,9])
# x0 = np.array([9,3])

A = np.zeros((n,n))
B = np.eye(n)
plant = LinearSystem(x0, np.zeros(2), A=A, B=B)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFaxes = np.array([1.0, 4.0])
CLFangle = 0.0
CLFcenter = np.zeros(2)
clf = QuadraticLyapunov.geometry2D(CLFaxes, CLFangle, CLFcenter, level=1, limits=limits)
Hv = clf.H

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFaxes = [2.0, 1.0]
CBFangle = 10.0
CBFcenter = np.array([0.0, 5.0])
cbf1 = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter, limits=limits)

CBFaxes = [1.0, 2.0]
CBFangle = -10.0
CBFcenter = np.array([6.0, 0.0])
cbf2 = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter, limits=limits)

cbfs = [cbf1, cbf2]
num_cbfs = len(cbfs)

''' --------------------------- Compatible controller --------------------------------- '''
T = 15
sample_time = 5e-3
controller = CompatibleQP(plant, clf, cbfs, 
                          alpha = [3.0, 10.0], beta = [10.0, 10.0], p = [1.0, 1.0], dt = sample_time, 
                          compatibilization=True)

''' ------------------------------ Configure plot ----------------------------------- '''
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

logs = { "sample_time": sample_time }