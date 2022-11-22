import numpy as np

from dynamic_systems import Periodic
from functions import canonical2D

######################################### Configure and create unicycle plant ##############################################
initial_state = [-4.1, 5.0]

g11 = np.array([ [ 1, 0 ], 
                 [ 0, 1 ] ])
g12 = np.array([ [ 0, -1 ], 
                 [ 1,  0 ] ])
g21 = np.zeros([2,2])
g22 = np.zeros([2,2])

G = np.array([  [g11, g21], 
                [g12, g22],  ])
F = np.array([  [1, 0],
                [1, 0] ])
P = np.array([  [np.pi/2, 0],
                [0      , 0] ])
plant = Periodic(initial_state = initial_state, initial_control = np.zeros(2), Gains=G, Frequencies = F, Phases = P)
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
cbf_lambda_x, cbf_lambda_y, cbf_angle = 1.0, 3.0, np.radians(0.0)
cbf_params1 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [ 0.0, 3.0 ] }

cbf_lambda_x, cbf_lambda_y, cbf_angle = 6.0, 1.0, np.radians(30.0)
cbf_params2 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [ 3.0, 3.0 ] }

cbf_lambda_x, cbf_lambda_y, cbf_angle = 4.0, 1.0, np.radians(30.0)
cbf_params3 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [ -3.0, -3.0 ] }
############################################################################################################################