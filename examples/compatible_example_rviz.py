import rospy
import numpy as np

from system_initialization import plant, clf, cbf, ref_clf
from controllers import CompatibleQPController
from graphical_simulation import SimulationRviz

try:
    # Create QP controller and graphical simulation.
    dt = 0.004
    qp_controller = CompatibleQPController(plant, clf, ref_clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 10.0], p = [1.0, 1.0], dt=dt)
    graphicalSimulation = SimulationRviz(plant, qp_controller.clf, cbf)

    rate = rospy.Rate(1/dt)
    while not rospy.is_shutdown():

        # Control
        u_control, upi_control = qp_controller.get_control()
        # upi_control = np.zeros(3)
        qp_controller.update_clf_dynamics(upi_control)
        
        # Send actuation commands
        plant.set_control(u_control) 
        plant.actuate(dt)

        # print("CBF = " + str(qp_controller.h))

        # print("Compatibility Barrier 1 = " + str(qp_controller.h_gamma1))
        # print("Compatibility Barrier 2 = " + str(qp_controller.h_gamma2))
        # print("Compatibility Barrier 3 = " + str(qp_controller.h_gamma3))

        # Draw graphical simulation elements
        graphicalSimulation.draw_trajectory()
        graphicalSimulation.draw_reference(qp_controller.clf.critical_point)
        graphicalSimulation.draw_clf()
        graphicalSimulation.draw_cbf()
        # graphicalSimulation.draw_invariance(qp_controller)

        rate.sleep()

except rospy.ROSInterruptException:
    pass