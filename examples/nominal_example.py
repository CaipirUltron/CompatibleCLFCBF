import rospy

from system_initialization import plant, clf, cbf, dynamicSimulation, graphicalSimulation, dt
from compatible_clf_cbf.controller import NominalQP

try:
    # Create QP controller
    qp_controller = NominalQP(plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0)

    rate = rospy.Rate(1/dt)
    while not rospy.is_shutdown():

        # Get simulation state
        state = dynamicSimulation.state()

        # Control
        u_control = qp_controller.get_control(state)

        # Send actuation commands 
        dynamicSimulation.send_control_inputs(u_control, dt)

        # Draw graphical simulation elements
        graphicalSimulation.draw_trajectory(state)
        graphicalSimulation.draw_clf(qp_controller.clf, state)
        graphicalSimulation.draw_cbf()

        rate.sleep()

except rospy.ROSInterruptException:
    pass