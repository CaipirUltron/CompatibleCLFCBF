import rospy
import math
import numpy as np
from compatible_clf_cbf.controller import QPController
from compatible_clf_cbf.dynamic_simulation import SimulateDynamics
from compatible_clf_cbf.graphical_simulation import GraphicalSimulation
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier, QuadraticFunction
    
# Simulation parameters
dt = .002
sim_freq = 1/dt
T = 20

# Define 2D plant and initial state
n = 3
f = ['0','0','0']
g1 = ['1','0','0']
g2 = ['0','1','0']
g3 = ['0','0','1']
g = [g1,g2,g3]
state_string = 'x1, x2, x3'
control_string = 'u1, u2, u3'
plant = AffineSystem(state_string, control_string, f, *g)

# Create random CLF and CBF
Hv = np.zeros([n,n])
Hh = np.zeros([n,n])
basis = QuadraticFunction.symmetric_basis(n)
for k in range(np.size(basis,0)):
    Hv = Hv + (2*np.random.rand())*basis[k]
    Hh = Hh + (2*np.random.rand())*basis[k]

eigHv, _ = np.linalg.eig(Hv)
eigHh, _ = np.linalg.eig(Hh)

print("Eigenvalues of Hv:" + str(eigHv))
print("Eigenvalues of Hh:" + str(eigHh))

rng = 10
x0 = rng*( 2*np.random.rand(n) - np.ones(n) )
p0 = rng*( 2*np.random.rand(n) - np.ones(n) )

clf = QuadraticLyapunov(state_string, Hv, x0)
cbf = QuadraticBarrier(state_string, Hh, p0)

# Create QP controller
qp_controller = QPController(plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0)

print("Pencil eigenvalues:" + str(qp_controller.pencil_char_roots))
print("det(H(\lambda)) = " + str(qp_controller.pencil_char))

print("Zeros = " + str(qp_controller.num_roots))
print("Poles = " + str(qp_controller.den_roots))