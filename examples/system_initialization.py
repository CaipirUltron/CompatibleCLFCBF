import numpy as np

from dynamic_systems import PolynomialSystem
from functions import canonical2D

######################################### Configure and create 2D plant ####################################################
initial_state = [5.1, 5.5]

G1 = np.array([[1,0],[0,1]])
G2 = np.array([[0,0],[0,0]])
G3 = np.array([[0,0],[0,0]])
G = [ G1, G2, G3 ]
plant = PolynomialSystem(initial_state = initial_state, initial_control = np.zeros(2), degree=1, G_list = G)
############################################################################################################################

############################################# Configure and create CLF #####################################################
clf_lambda_x, clf_lambda_y, clf_angle = 6.0, 1.0, np.radians(-45.0)
clf_params = {
    "Hv": canonical2D([ clf_lambda_x , clf_lambda_y ], clf_angle),
    "x0": [ 0.0, 0.0 ] }
############################################################################################################################

######################################## Configure and create reference CLF ################################################
ref_clf_lambda_x, ref_clf_lambda_y, ref_clf_angle = 6.0, 1.0, np.radians(-45.0)
ref_clf_params = {
    "Hv": canonical2D([ ref_clf_lambda_x , ref_clf_lambda_y ], ref_clf_angle),
    "x0": [ 0.0, 0.0 ] }
############################################################################################################################

############################################## Configure and create CBF ####################################################
# xaxis_length, yaxis_length, cbf_angle = 3.0, 1.0, np.radians(0.0)
# cbf_params = {
#     "Hh": canonical2D([ 1/(xaxis_length**2), 1/(yaxis_length**2) ], cbf_angle),
#     "p0": [ 0.0, 3.0 ]
# }
cbf_lambda_x, cbf_lambda_y, cbf_angle = 1.0, 3.0, np.radians(0.0)
cbf_params1 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [ 0.0, 3.0 ] }

cbf_lambda_x, cbf_lambda_y, cbf_angle = 6.0, 1.0, np.radians(30.0)
cbf_params2 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [ 3.0, 3.0 ] }
############################################################################################################################