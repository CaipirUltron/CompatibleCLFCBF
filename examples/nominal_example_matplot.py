import rospy

from system_initialization import plant, clf, cbf
from compatible_clf_cbf.controller import NominalQP
from compatible_clf_cbf.graphical_simulation import SimulationMatplot

try:
    # Create QP controller and graphical simulation.
    qp_controller = NominalQP(plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0)
    graphicalSimulation = SimulationMatplot(plant, qp_controller.clf, cbf)

    dt = 0.004
    rate = rospy.Rate(1/dt)
    while not rospy.is_shutdown():

        # Control
        u_control = qp_controller.get_control()

        # Send actuation commands 
        plant.set_control(u_control) 
        plant.actuate(dt)

        # Draw graphical simulation elements
        graphicalSimulation.draw_trajectory()
        graphicalSimulation.draw_clf()
        graphicalSimulation.draw_cbf()

        rate.sleep()

except rospy.ROSInterruptException:
    pass