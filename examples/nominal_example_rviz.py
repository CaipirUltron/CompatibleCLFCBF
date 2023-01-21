import rospy

from examples.integrator_nominalQP import plant, clf, cbf
from controllers import NominalQP
from graphics import SimulationRviz

# clf = PolynomialFunction(*initial_state, degree = 2)
clf = QuadraticLyapunov(*initial_state, hessian = clf_params["Hv"], critical = clf_params["x0"])
# clf_gaussian_component = Gaussian(*initial_state, constant=3.0, mean=[ 0.0, 3.0 ], shape=np.diag([15, 1]))

ref_clf = QuadraticLyapunov(*initial_state, hessian = ref_clf_params["Hv"], critical = ref_clf_params["x0"])
cbf = QuadraticBarrier(*initial_state, hessian = cbf_params["Hh"], critical = cbf_params["p0"])
############################################################################################################################

#################################################### SoS Controller ########################################################
controller = SoSController( plant, clf, cbf )

try:
    # Create QP controller and graphical simulation.
    qp_controller = NominalQP(plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0)
    graphicalSimulation = SimulationRviz(plant, qp_controller.clf, cbf)

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