import math
import numpy as np

from compatible_clf_cbf.dynamic_systems import Quadratic, QuadraticLyapunov, QuadraticBarrier, LinearSystem, CassiniOval

######################################### Configure and create 2D plant ####################################################
initial_state = [2.0, 3.5]
# plant = Integrator(initial_state, initial_control = np.zeros(2))
plant = LinearSystem(initial_state, initial_control = np.zeros(2), A = np.zeros([2,2]), B = np.array([[0,-1],[1,0]]))
############################################################################################################################

############################################# Configure and create CLF #####################################################
clf_lambda_x, clf_lambda_y, clf_angle = 6.0, 1.0, math.radians(0.0)
clf_params = {
    "Hv": Quadratic.canonical2D([ clf_lambda_x , clf_lambda_y ], clf_angle),
    "x0": [ 0.0, 0.0 ] }
clf = QuadraticLyapunov(init_value = initial_state, hessian = clf_params["Hv"], critical = clf_params["x0"])
############################################################################################################################

######################################## Configure and create reference CLF ################################################
ref_clf_lambda_x, ref_clf_lambda_y, ref_clf_angle = 6.0, 1.0, math.radians(0.0)
ref_clf_params = {
    "Hv": Quadratic.canonical2D([ ref_clf_lambda_x , ref_clf_lambda_y ], ref_clf_angle),
    "x0": [ 0.0, 0.0 ] }
ref_clf = QuadraticLyapunov(init_value = initial_state, hessian = ref_clf_params["Hv"], critical = ref_clf_params["x0"])
############################################################################################################################

############################################## Configure and create CBF ####################################################
# xaxis_length, yaxis_length, cbf_angle = 3.0, 1.0, math.radians(0.0)
# cbf_params = {
#     "Hh": Quadratic.canonical2D([ 1/(xaxis_length**2), 1/(yaxis_length**2) ], cbf_angle),
#     "p0": [ 0.0, 3.0 ]
# }
cbf_lambda_x, cbf_lambda_y, cbf_angle = 1.0, 3.0, math.radians(0.0)
cbf_params = {
    "Hh": Quadratic.canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [ 0.0, 3.0 ] }
cbf = QuadraticBarrier(init_value = initial_state, hessian = cbf_params["Hh"], critical = cbf_params["p0"])

# a, b = 1.0, 2.0
# angle = 0.0
# cbf = CassiniOval(a, b, angle, init_value = initial_state)
############################################################################################################################