import rospy

from system_initialization import plant, clf, cbf, ref_clf
from compatible_clf_cbf.controller import NewQPController
from compatible_clf_cbf.graphical_simulation import SimulationRviz

try:
    # Create QP controller and graphical simulation.
    qp_controller = NewQPController(plant, clf, ref_clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 10.0], p = [1.0, 1.0])
    graphicalSimulation = SimulationRviz(plant, qp_controller.clf, cbf)

    dt = 0.004
    rate = rospy.Rate(1/dt)
    while not rospy.is_shutdown():

        # Control
        u_control, upi_control = qp_controller.get_control()
        qp_controller.update_clf_dynamics(upi_control)
        
        # Send actuation commands
        plant.set_control(u_control) 
        plant.actuate(dt)

        # print("Compatibility Barrier 2 = " + str(qp_controller.h_gamma2))

        # Draw graphical simulation elements
        graphicalSimulation.draw_trajectory()
        graphicalSimulation.draw_reference(qp_controller.clf.critical_point)
        graphicalSimulation.draw_clf()
        graphicalSimulation.draw_cbf()
        # graphicalSimulation.draw_invariance(qp_controller)

        rate.sleep()

except rospy.ROSInterruptException:
    pass