import numpy as np

from common import is_controllable
from controllers import CompatibleQP
from dynamic_systems import LinearSystem, DriftLess
from functions import QuadraticLyapunov, QuadraticBarrier

''' --------------------------------- Define LTI system ------------------------------------- '''
x0 = np.array([1,8])

# A = np.array([[0,0],
#               [0,0]])
# B = np.array([[1,0],
#               [0,1]])

A = np.array([[-2, 0],
              [ 0,-2]])
B = np.array([[1,0],
              [0,1]])

# A = np.array([[ 0, 1],
#               [-1,-1]])

# A = np.array([[ 0,-1],
#               [-1,-1]])
# B = np.array([[0],
#               [1]])

n = A.shape[0]
m = B.shape[1]

msg = f"LTI plant (A,B) is "
if is_controllable(A,B):
    eigA = np.linalg.eigvals(A)
    print(msg+f'controllable with λ(A) = \n{eigA}')
else: 
    print(msg+'not controllable.')

plant = LinearSystem(x0, np.zeros(m), A=A, B=B)

''' ------------------------ Define CLF (varying Hessian eigenvalues) ----------------------- '''
CLFaxes = np.array([1.0, 4.0])
CLFangle = 0.0
CLFcenter = np.zeros(2)

clf = QuadraticLyapunov.geometry2D(CLFaxes, CLFangle, CLFcenter, level=1)

''' ------------------------ Define CBF (varying Hessian eigenvalues) ----------------------- '''
CBFaxes = [4.0, 1.0]
CBFangle = 10.0
CBFcenter = np.array([0.0, 5.0])
cbf1 = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter)

CBFaxes = [1.0, 3.0]
CBFangle = -10.0
CBFcenter = np.array([6.0, 0.0])
cbf2 = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter)

CBFaxes = [1.0, 2.0]
CBFangle = -10.0
CBFcenter = np.array([-6.0, 2.0])
cbf3 = QuadraticBarrier.geometry2D(CBFaxes, CBFangle, CBFcenter)

# cbfs = []
# cbfs = [cbf1]
cbfs = [cbf1, cbf2, cbf3]
num_cbfs = len(cbfs)

''' --------------------------- Compatible controller --------------------------------- '''
T = 10
sample_time = 5e-3
controller = CompatibleQP(plant, clf, cbfs, 
                          alpha = [1.0, 1.0], beta = 1.0, p = [1.0, 1.0], dt = sample_time, 
                          compatibilization=True,
                          active=True,
                          verbose=True)

''' ------------------------------ Configure plot ----------------------------------- '''
plot_config = {
    "xlimits": (-9,9),
    "ylimits": (-5,9),
    "drawlevel": True,
    "resolution": 50,
    "fps":60,
    "equilibria": False,
    }

logs = { "sample_time": sample_time }