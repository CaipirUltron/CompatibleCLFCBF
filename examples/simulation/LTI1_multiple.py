'''
Simulation example. Has to define the following objects:
1) plant - system dynamics (LinearSystem, DriftLess)
2) clf: QuadraticLyapunov - CLF moding the stabilization requirement
3) cbfs: list[QuadraticBarrier] - list of quadratic CBFs modeling the safety requirements
'''

import numpy as np

from common import is_controllable
from dynamic_systems import LinearSystem, DriftLess
from functions import QuadraticLyapunov, QuadraticBarrier

limits = 9*np.array((-1,1,-1,1))

''' --------------------------------- Define LTI system ------------------------------------- '''
x0 = np.array([1,8])

A = np.array([[-2, 0],
              [ 0,-2]])
B = np.array([[1,0],
              [0,1]])

n = A.shape[0]
m = B.shape[1]

msg = f"LTI plant (A,B) is "
if is_controllable(A,B):
    eigA = np.linalg.eigvals(A)
    print(msg+f'controllable with λ(A) = \n{eigA}')
else: 
    print(msg+'not controllable.')

plant = LinearSystem(A=A, B=B, state = x0)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFaxes = np.array([1.0, 4.0])
CLFangle = 0.0
CLFcenter = np.zeros(2)

clf = QuadraticLyapunov.geometry2D(CLFaxes, CLFangle, CLFcenter, level=1, limits=limits)

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFaxes = [4.0, 1.0]
CBFangle = 10.0
CBFcenter = np.array([0.0, 5.0])
cbf1 = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter, limits=limits)

CBFaxes = [1.0, 3.0]
CBFangle = -10.0
CBFcenter = np.array([6.0, 0.0])
cbf2 = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter, limits=limits)

CBFaxes = [1.0, 2.0]
CBFangle = -10.0
CBFcenter = np.array([-6.0, 2.0])
cbf3 = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter, limits=limits)

# cbfs = []
cbfs = [cbf3]
# cbfs = [cbf1, cbf2, cbf3]
num_cbfs = len(cbfs)

''' ------------------------------ Configure plot ----------------------------------- '''
plot_config = {
    "xlimits": (-9,9),
    "ylimits": (-5,9),
    "drawlevel": True,
    "resolution": 50,
    "fps":60,
    "equilibria": False,
    }