import numpy as np
from functions import canonical2D
from functions import QuadraticLyapunov, QuadraticBarrier
from dynamic_systems import LinearSystem
from controllers import CompatibleQP

''' ---------------------------------------- LTI plant -------------------------------------------- '''
initial_state = [4.2, 5.0]

# G1 = np.array([[1,0],[0,1]])
# G2 = np.array([[0,0],[0,0]])
# G3 = np.array([[0,0],[0,0]])
# G = [ G1 ]
# plant = PolynomialSystem(initial_state = initial_state, initial_control = np.zeros(2), degree=0, G_list = G)

A = np.array([[1,0],
              [0,1]])
B = np.array([[1,0],
              [0,1]])

plant = LinearSystem(initial_state = initial_state, initial_control = np.zeros(2), A=A, B=B)


''' ------------------------------------------- CLF ----------------------------------------------- '''
clf_lambda_x, clf_lambda_y, clf_angle = 6.0, 1.0, np.radians(-45.0)
clf_params = {
    "Hv": canonical2D([ clf_lambda_x , clf_lambda_y ], clf_angle),
    "x0": [ 0.0, 0.0 ] }

clf = QuadraticLyapunov(*initial_state, hessian = clf_params["Hv"], critical = clf_params["x0"])

''' ------------------------------------------- CBFs ---------------------------------------------- '''
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

cbf1 = QuadraticBarrier(*initial_state, hessian = cbf_params1["Hh"], critical = cbf_params1["p0"])
cbf2 = QuadraticBarrier(*initial_state, hessian = cbf_params2["Hh"], critical = cbf_params2["p0"])
cbf3 = QuadraticBarrier(*initial_state, hessian = cbf_params3["Hh"], critical = cbf_params3["p0"])

cbfs = [cbf1, cbf2, cbf3]

''' -------------------------------------------- Controller ------------------------------------------------------ '''
sample_time = .001
controller = CompatibleQP(plant, clf, cbfs, alpha = [10.0, 1.0], beta = [10.0, 10.0], p = [1.0, 1.0], dt = sample_time)

####################################################### Configure plot #####################################################
xlimits, ylimits = [-8, 8], [-8, 8]
plot_config = {
    "figsize": (5,5),
    "gridspec": (1,1,1),
    "widthratios": [1],
    "heightratios": [1],
    "axeslim": tuple(xlimits+ylimits),
    "path_length": 10,
    "numpoints": 1000,
    "drawlevel": True,
    "resolution": 50,
    "fps":120,
    "pad":2.0,
    "equilibria": True
}

logs = { "sample_time": sample_time }