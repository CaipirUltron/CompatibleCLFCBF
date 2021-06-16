import rospy, math
import numpy as np

from compatible_clf_cbf.controller import QPController
from compatible_clf_cbf.dynamic_simulation import SimulateDynamics
from compatible_clf_cbf.graphical_simulation import SimulationRviz
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier, QuadraticFunction

try:
    ######################################### Configure and create 2D plant ####################################################
    system = {
        "f": ['0','0'],
        "g": [['1','0'],['0','1']],
        "state_string": 'x1, x2, ',
        "control_string": 'u1, u2, ',
        "initial_state": np.array([ -10.0, 5.0 ])
    }
    plant = AffineSystem(system["state_string"], system["control_string"], system["f"], *system["g"])
    ############################################################################################################################


    ############################################# Configure and create CLF #####################################################
    clf_lambda_x, clf_lambda_y, clf_angle = 1.0, 6.0, math.radians(-45.0)
    clf_config = {
        "Hv": QuadraticFunction.canonical2D(np.array([ clf_lambda_x , clf_lambda_y ]), clf_angle),
        "x0": np.array([ 0, 0 ]),
    }
    clf = QuadraticLyapunov(system["state_string"], hessian = clf_config["Hv"], critical = clf_config["x0"])
    ############################################################################################################################


    ######################################## Configure and create reference CLF ################################################
    ref_clf_lambda_x, ref_clf_lambda_y, ref_clf_angle = 1.0, 6.0, math.radians(-45.0)
    ref_clf_config = {
        "Hv": QuadraticFunction.canonical2D(np.array([ ref_clf_lambda_x , ref_clf_lambda_y ]), ref_clf_angle),
        "x0": np.array([ 0, 0 ]),
    }
    ref_clf = QuadraticLyapunov(system["state_string"], hessian = ref_clf_config["Hv"], critical = ref_clf_config["x0"])
    ############################################################################################################################


    ############################################## Configure and create CBF ####################################################
    xaxis_length, yaxis_length, cbf_angle = 3.0, 1.0, math.radians(60.0)
    cbf_config = {
        "Hh": QuadraticFunction.canonical2D(np.array([ 1/(xaxis_length**2), 1/(yaxis_length**2) ]), cbf_angle),
        "p0": np.array([ -3.0, 3.0 ])
    }
    cbf = QuadraticBarrier(system["state_string"], hessian = cbf_config["Hh"], critical = cbf_config["p0"])
    ############################################################################################################################

    # Create QP controller.
    qp_controller = QPController(plant, clf, ref_clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 10.0], p = [10.0, 10.0])

    # Initialize simulation objects and main loop.
    dynamicSimulation = SimulateDynamics(plant, system["initial_state"])
    graphicalSimulation = SimulationRviz(clf, cbf)

    dt = .005
    rate = rospy.Rate(1/dt)
    while not rospy.is_shutdown():

        # Get simulation state
        state = dynamicSimulation.state()

        # Control
        control = qp_controller.compute_control(state)

        # Send actuation commands 
        dynamicSimulation.send_control_inputs(control, dt)

        # Draw graphical simulation elements
        graphicalSimulation.draw_trajectory(state)
        graphicalSimulation.draw_reference(qp_controller.clf.critical_point)
        graphicalSimulation.draw_clf(qp_controller.clf, state)
        graphicalSimulation.draw_cbf()
        graphicalSimulation.draw_invariance(qp_controller)

        rate.sleep()

except rospy.ROSInterruptException:
    pass