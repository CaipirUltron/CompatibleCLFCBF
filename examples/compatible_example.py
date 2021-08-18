import rospy

from system_initialization import plant, clf, cbf, ref_clf, dynamicSimulation, graphicalSimulation, dt
from compatible_clf_cbf.controller import NewQPController

try:
    # Create QP controller
    qp_controller = NewQPController(plant, clf, ref_clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 10.0], p = [1.0, 1.0])

    rate = rospy.Rate(1/dt)
    while not rospy.is_shutdown():

        # Get simulation state
        state = dynamicSimulation.state()

        # Control
        u_control, upi_control = qp_controller.get_control(state)
        qp_controller.update_clf_dynamics(upi_control)

        # Send actuation commands 
        dynamicSimulation.send_control_inputs(u_control, dt)

        # Draw graphical simulation elements
        graphicalSimulation.draw_trajectory(state)
        graphicalSimulation.draw_reference(qp_controller.clf.critical_point)
        graphicalSimulation.draw_clf(qp_controller.clf, state)
        graphicalSimulation.draw_cbf()
        # graphicalSimulation.draw_invariance(qp_controller)

        rate.sleep()

except rospy.ROSInterruptException:
    pass