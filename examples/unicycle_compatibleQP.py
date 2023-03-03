import numpy as np

from dynamic_systems import Unicycle
from functions import canonical3D
from functions import QuadraticLyapunov, QuadraticBarrier
from controllers import CompatibleQPController

######################################### Configure and create unicycle plant ##############################################
initial_state = [ -4.1, 5.0, np.radians(0.0) ]
plant = Unicycle(initial_state = initial_state, initial_control = np.zeros(2), radius = 0.5)
############################################################################################################################

############################################# Configure and create CLF #####################################################
clf_lambda_x, clf_lambda_y, clf_lambda_z = 6.0, 1.0, 1.0 
clf_angle, clf_axis = np.radians(0.0), np.array([0.0, 0.0, 1.0])
clf_params = {
    "Hv": canonical3D([ clf_lambda_x , clf_lambda_y, clf_lambda_z ], clf_angle, clf_axis),
    "x0": [ 0.0, 0.0, 0.0 ] }
############################################################################################################################

######################################## Configure and create reference CLF ################################################
ref_clf_lambda_x, ref_clf_lambda_y, ref_clf_lambda_z = 6.0, 1.0, 1.0
ref_clf_angle, ref_clf_axis = np.radians(0.0), np.array([0.0, 0.0, 1.0])
ref_clf_params = {
    "Hv": canonical3D([ ref_clf_lambda_x , ref_clf_lambda_y, ref_clf_lambda_z ], ref_clf_angle, ref_clf_axis),
    "x0": [ 0.0, 0.0, 0.0 ] }
############################################################################################################################

############################################## Configure and create CBF ####################################################
cbf_lambda_x, cbf_lambda_y, cbf_lambda_z = 1.0, 3.0, 0.1
cbf_angle, cbf_axis = np.radians(0.0), np.array([0.0, 0.0, 1.0])
cbf_params1 = {
    "Hh": canonical3D([ cbf_lambda_x , cbf_lambda_y, cbf_lambda_z ], cbf_angle, cbf_axis),
    "p0": [ 0.0, 3.0, 0.0 ] }

cbf_lambda_x, cbf_lambda_y, cbf_lambda_z = 6.0, 1.0, 0.1
cbf_angle, cbf_axis = np.radians(30.0), np.array([0.0, 0.0, 1.0])
cbf_params2 = {
    "Hh": canonical3D([ cbf_lambda_x , cbf_lambda_y, cbf_lambda_z ], cbf_angle, cbf_axis),
    "p0": [ 3.0, 3.0, 0.0 ] }

cbf_lambda_x, cbf_lambda_y, cbf_lambda_z = 4.0, 1.0, 0.1
cbf_angle, cbf_axis = np.radians(30.0), np.array([0.0, 0.0, 1.0])
cbf_params3 = {
    "Hh": canonical3D([ cbf_lambda_x , cbf_lambda_y, cbf_lambda_z ], cbf_angle, cbf_axis),
    "p0": [ -3.0, -3.0, 0.0] }
############################################################################################################################

########################################## Define quadratic Lyapunov and barriers ##########################################
clf = QuadraticLyapunov(*initial_state, hessian = clf_params["Hv"], critical = clf_params["x0"])
ref_clf = QuadraticLyapunov(*initial_state, hessian = ref_clf_params["Hv"], critical = ref_clf_params["x0"])

cbf1 = QuadraticBarrier(*initial_state, hessian = cbf_params1["Hh"], critical = cbf_params1["p0"])
cbf2 = QuadraticBarrier(*initial_state, hessian = cbf_params2["Hh"], critical = cbf_params2["p0"])
cbf3 = QuadraticBarrier(*initial_state, hessian = cbf_params3["Hh"], critical = cbf_params3["p0"])

cbfs = [cbf1, cbf2, cbf3]

#################################################### Define controllers ####################################################
sample_time = .005
controller = CompatibleQPController(plant, clf, ref_clf, cbfs, gamma = [1.0, 10.0], alpha = [1.0, 10.0], p = [1.0, 1.0], dt = sample_time)