'''
Simulation example. Has to define the following objects:
1) plant - system dynamics (LinearSystem, DriftLess)
2) clf: QuadraticLyapunov - CLF moding the stabilization requirement
3) cbfs: list[QuadraticBarrier] - list of quadratic CBFs modeling the safety requirements
'''
import numpy as np

from common import is_controllable, rot3D
from dynamic_systems import LinearSystem
from functions import QuadraticLyapunov, QuadraticBarrier

limits = 9*np.array((-1,1,-1,1))

''' --------------------------------- Define LTI system ------------------------------------- '''
x0 = np.array([-1, 0.5, 6])

# A = -0.01*np.eye(3)
# A = 0.1*np.array([[ 0, 1, 0], [ -1,-1, 0], [ 0, 0,-1]])
A = np.zeros([3,3])

B = np.array([[1,0,-2],
              [0,1,0],
              [-2,0,1]])

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
CLFaxes = np.array([2.0, 2.0, 6.0])
CLFrot = rot3D(theta=0.0, axis=[1.0, 0.0, 0.0])
CLFcenter = np.zeros(n)
clf = QuadraticLyapunov.geometry(CLFaxes, CLFrot, CLFcenter, level=1)

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFaxes = [2.0, 1.0, 1.0]
CBFrot = rot3D(theta=0.0, axis=[0.0, 0.0, 1.0])
CBFcenter = np.array([-1.0, 0.0, 3.0])
cbf1 = QuadraticBarrier.geometry(CBFaxes, CBFrot, CBFcenter)

CBFaxes = [2.0, 1.0, 1.0]
CBFrot = rot3D(theta=0.0, axis=[1.0, 0.0, 1.0])
CBFcenter = np.array([1.0, 0.0, 3.0])
cbf2 = QuadraticBarrier.geometry(CBFaxes, CBFrot, CBFcenter)

CBFaxes = [1.0, 2.0, 3.0]
CBFrot = rot3D(theta=10.0, axis=[1.0, 0.0, 1.0])
CBFcenter = np.array([-6.0, 2.0, -1.0])
cbf3 = QuadraticBarrier.geometry(CBFaxes, CBFrot, CBFcenter)

# cbfs = []
# cbfs = [cbf1]
cbfs = [cbf1, cbf2]
num_cbfs = len(cbfs)

''' ------------------------------ Configure plot ----------------------------------- '''
plot_config = {
    "xlimits": (-2.0, 2.0),
    "ylimits": (-2.0, 2.0),
    "zlimits": (0,5),
    "drawlevel": True,
    "resolution": 50,
    "fps":60,
    "equilibria": False,
    "arrows": True
    }