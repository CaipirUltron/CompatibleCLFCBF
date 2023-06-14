import numpy as np

from dynamic_systems import Unicycle
from functions import canonical2D
from functions import QuadraticLyapunov, QuadraticBarrier
from controllers import NominalQP

######################################### Configure and create unicycle plant ##############################################
distance = 0.5
initial_state = [ -4.1, 5.0, np.radians(0.0) ]
plant = Unicycle(initial_state = initial_state, initial_control = np.zeros(2), distance = distance)
############################################################################################################################

############################################# Configure and create CLF #####################################################
clf_lambda_x, clf_lambda_y = 6.0, 1.0
clf_angle = np.radians(0.0)
clf_params = {
    "Hv": canonical2D([ clf_lambda_x , clf_lambda_y ], clf_angle),
    "x0": [ 0.0, 0.0 ] }
############################################################################################################################

######################################## Configure and create reference CLF ################################################
ref_clf_lambda_x, ref_clf_lambda_y = 6.0, 1.0
ref_clf_angle = np.radians(0.0)
ref_clf_params = {
    "Hv": canonical2D([ ref_clf_lambda_x , ref_clf_lambda_y ], ref_clf_angle),
    "x0": [ 0.0, 0.0 ] }
############################################################################################################################

############################################## Configure and create CBF ####################################################
cbf_lambda_x, cbf_lambda_y = 1.0, 3.0
cbf_angle = np.radians(0.0)
cbf_params1 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [ 0.0, 3.0 ] }

cbf_lambda_x, cbf_lambda_y = 6.0, 1.0
cbf_angle = np.radians(30.0)
cbf_params2 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [ 3.0, 3.0 ] }

cbf_lambda_x, cbf_lambda_y = 4.0, 1.0
cbf_angle = np.radians(30.0)
cbf_params3 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [ -3.0, -3.0 ] }
############################################################################################################################

########################################## Define quadratic Lyapunov and barriers ##########################################
collapsed_state = initial_state[:2]
clf = QuadraticLyapunov(*collapsed_state, hessian = clf_params["Hv"], critical = clf_params["x0"])
ref_clf = QuadraticLyapunov(*collapsed_state, hessian = ref_clf_params["Hv"], critical = ref_clf_params["x0"])

cbf1 = QuadraticBarrier(*collapsed_state, hessian = cbf_params1["Hh"], critical = cbf_params1["p0"])
cbf2 = QuadraticBarrier(*collapsed_state, hessian = cbf_params2["Hh"], critical = cbf_params2["p0"])
cbf3 = QuadraticBarrier(*collapsed_state, hessian = cbf_params3["Hh"], critical = cbf_params3["p0"])

cbfs = [cbf1, cbf2, cbf3]

#################################################### Define controllers ####################################################
sample_time = .005
controller = NominalQP(plant, clf, cbfs, gamma = 1.0, alpha = 1.0, p = 10.0)