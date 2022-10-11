import numpy as np

from dynamic_systems import PolynomialSystem
from functions import canonical3D

######################################### Configure and create 2D plant ####################################################
initial_state = [0.1, 0.1, 5.5]

G1 = np.eye(3)
G2 = np.zeros([3,3])
G3 = np.zeros([3,3])
G4 = np.zeros([3,3])
G = [ G1, G2, G3, G4 ]
plant = PolynomialSystem(initial_state = initial_state, initial_control = np.zeros(3), degree=1, G_list = G)
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
# xaxis_length, yaxis_length, cbf_angle = 3.0, 1.0, np.radians(0.0)
# cbf_params = {
#     "Hh": canonical2D([ 1/(xaxis_length**2), 1/(yaxis_length**2) ], cbf_angle),
#     "p0": [ 0.0, 3.0 ]
# }
cbf_lambda_x, cbf_lambda_y, cbf_lambda_z = 1.0, 3.0, 1.0
cbf_angle, cbf_axis = np.radians(0.0), np.array([0.0, 0.0, 1.0])
cbf_params1 = {
    "Hh": canonical3D([ cbf_lambda_x , cbf_lambda_y, cbf_lambda_z ], cbf_angle, cbf_axis),
    "p0": [ 0.0, 3.0, 0.0 ] }

cbf_lambda_x, cbf_lambda_y, cbf_lambda_z = 6.0, 1.0, 1.0
cbf_angle, cbf_axis = np.radians(30.0), np.array([0.0, 0.0, 1.0])
cbf_params2 = {
    "Hh": canonical3D([ cbf_lambda_x , cbf_lambda_y, cbf_lambda_z ], cbf_angle, cbf_axis),
    "p0": [ 3.0, 3.0, 0.0 ] }

cbf_lambda_x, cbf_lambda_y, cbf_lambda_z = 4.0, 1.0, 1.0
cbf_angle, cbf_axis = np.radians(30.0), np.array([0.0, 0.0, 1.0])
cbf_params3 = {
    "Hh": canonical3D([ cbf_lambda_x , cbf_lambda_y, cbf_lambda_z ], cbf_angle, cbf_axis),
    "p0": [ -3.0, -3.0, 0.0] }
############################################################################################################################