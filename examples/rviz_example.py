
import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from compatible_clf_cbf.controller import QPController
from compatible_clf_cbf.dynamic_simulation import SimulateDynamics
from compatible_clf_cbf.graphical_simulation import SimulationRviz
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier, QuadraticFunction

try:
    # Simulation parameters
    dt = .005
    sim_freq = 1/dt
    T = 20

    # Define 2D plant and initial state
    f = ['0','0']
    g1 = ['1','0']
    g2 = ['0','1']
    g = [g1,g2]
    state_string = 'x1, x2, '
    control_string = 'u1, u2, '
    plant = AffineSystem(state_string, control_string, f, *g)

    # Define initial state for plant simulation
    x_init, y_init = 0.1, 5
    initial_state = np.array([x_init,y_init])

    # Create CLF
    lambdav_x, lambdav_y = 2.0, 1.0
    CLFangle = 0.0
    x0 = np.array([0,0])

    CLFeigen = np.array([ lambdav_x , lambdav_y ])
    Hv = QuadraticFunction.canonical2D(CLFeigen, CLFangle)
    clf = QuadraticLyapunov(state_string, Hv, x0)

    # Create CBF
    xaxis_length, yaxis_length = 2.0, 1.0
    CBFangle = math.radians(0.0)
    p0 = np.array([0,3])

    lambdah_x, lambdah_y = 1/xaxis_length**2, 1/yaxis_length**2
    CBFeigen = np.array([ lambdah_x , lambdah_y ])
    Hh = QuadraticFunction.canonical2D(CBFeigen, CBFangle)
    cbf = QuadraticBarrier(state_string, Hh, p0)

    # Create QP controller
    init_eig = [ 1.0, 2.0 ]   
    # init_eig = CLFeigen
    qp_controller = QPController(plant, clf, cbf, gamma = [1.0, 10.0], alpha = [1.0, 1.0], p = [10.0, 10.0], init_eig = init_eig)

    # Show initial plot of f(\lambda)
    print("Pencil eigenvalues:" + str(qp_controller.pencil_char_roots))
    print("Critical:" + str(qp_controller.critical_points))
    print("Critical values:" + str(qp_controller.critical_values))

    fig = plt.figure()
    axes_lim = (0, 20.0, 0, 300.0)
    ax = plt.axes(xlim=axes_lim[0:2], ylim=axes_lim[2:4])
    ax.set_title('CLF-CBF QP-based Control')
    lamb = np.arange(axes_lim[0], axes_lim[1], 0.01)
    fvalues = qp_controller.fvalues(lamb)
    ax.plot(lamb, fvalues, zorder=100, color='red')
    plt.show()

    # Initialize simulation object
    dynamicSimulation = SimulateDynamics(plant, initial_state)
    graphicalSimulation = SimulationRviz(clf, cbf)

    # Main loop
    rate = rospy.Rate(sim_freq)
    while not rospy.is_shutdown():

        # Get simulation state
        state = dynamicSimulation.state()

        # Control
        lambda_control, delta_pi = qp_controller.compute_lambda_control()
        qp_controller.update_clf_dynamics(lambda_control)
        control, delta = qp_controller.compute_control(state)

        print("Pencil eigenvalues:" + str(qp_controller.pencil_char_roots))
        print("Critical:" + str(qp_controller.critical_points))
        print("Critical values:" + str(qp_controller.critical_values))

        # Send actuation commands 
        dynamicSimulation.send_control_inputs(control, dt)

        # Draw graphical simulation elements
        graphicalSimulation.draw_trajectory(state)
        graphicalSimulation.draw_reference(qp_controller.clf.critical_point)
        graphicalSimulation.draw_clf(qp_controller.clf, state)
        graphicalSimulation.draw_cbf()

        rate.sleep()

except rospy.ROSInterruptException:
    pass