import json
import numpy as np
from functions import canonical2D
from functions import QuadraticLyapunov, QuadraticBarrier
from dynamic_systems import PolynomialSystem
from controllers import CompatiblePF, SplinePath

######################################### Configure and create 2D plant ####################################################
initial_state = [-7, -5.0]

G1 = np.array([[1,0],[0,1]])
G2 = np.array([[0,0],[0,0]])
G3 = np.array([[0,0],[0,0]])
G = [ G1 ]
plant = PolynomialSystem(initial_state = initial_state, initial_control = np.zeros(2), degree=0, G_list = G)

# ------------------------------------------- Configure and create path -----------------------------------------------------
# Load spline
path_name = "graphics/splines/path.json"
with open(path_name,'r') as file:
    print("Loading: " + path_name)
    path_params = json.load(file)
    path_params["points"] = np.array( path_params["points"] )

path = SplinePath(params=path_params, init_path_state=[np.deg2rad(0)])

# ------------------------------------------- Configure and create CLF -----------------------------------------------------
clf_lambda_x, clf_lambda_y, clf_angle = 1.0, 4.0, np.radians(0.0)
clf_params = {
    "Hv": canonical2D([ clf_lambda_x , clf_lambda_y ], clf_angle),
    "x0": [ 0.0, 0.0 ] }

# ---------------------------------------- Configure and create reference CLF ----------------------------------------------
ref_clf_lambda_x, ref_clf_lambda_y, ref_clf_angle = 1.0, 4.0, np.radians(0.0)
ref_clf_params = {
    "Hv": canonical2D([ ref_clf_lambda_x , ref_clf_lambda_y ], ref_clf_angle),
    "x0": [ 0.0, 0.0 ] }

############################################## Configure and create CBFs ###################################################
cbf_lambda_x, cbf_lambda_y, cbf_angle = 4.0, 1.0, np.radians(0.0)
cbf_params1 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [-4.0, 0.0] }

cbf_lambda_x, cbf_lambda_y, cbf_angle = 4.0, 1.0, np.radians(0.0)
cbf_params2 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [0.0, 0.0] }

cbf_lambda_x, cbf_lambda_y, cbf_angle = 4.0, 1.0, np.radians(0.0)
cbf_params3 = {
    "Hh": canonical2D([ cbf_lambda_x , cbf_lambda_y ], cbf_angle),
    "p0": [4.0, 0.0] }
############################################################################################################################

########################################## Define quadratic Lyapunov and barriers ##########################################
clf = QuadraticLyapunov(*initial_state, hessian = clf_params["Hv"], critical = clf_params["x0"])
ref_clf = QuadraticLyapunov(*initial_state, hessian = ref_clf_params["Hv"], critical = ref_clf_params["x0"])

cbf1 = QuadraticBarrier(*initial_state, hessian = cbf_params1["Hh"], critical = cbf_params1["p0"])
cbf2 = QuadraticBarrier(*initial_state, hessian = cbf_params2["Hh"], critical = cbf_params2["p0"])
cbf3 = QuadraticBarrier(*initial_state, hessian = cbf_params3["Hh"], critical = cbf_params3["p0"])

cbfs = [cbf1, cbf2, cbf3]

############################################## Configure and create controllers ############################################
sample_time = .001
controller = CompatiblePF(path, plant, clf, ref_clf, cbfs, alpha = [10.0, 1.0], beta = [10.0, 10.0], p = [50.0, 1.0], dt = sample_time)

####################################################### Configure plot #####################################################
xlimits, ylimits = [-10, 10], [-10, 10]
plot_config = {
    "figsize": (5,5),
    "gridspec": (1,1,1),
    "widthratios": [1],
    "heightratios": [1],
    "axeslim": tuple(xlimits+ylimits),
    "path_length": 10,
    "numpoints": 1000,
    "drawlevel": True,
    "resolution": 100,
    "fps":60,
    "pad":2.0
}