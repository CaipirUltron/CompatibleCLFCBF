import rospy
import numpy as np

from compatible_clf_cbf.controller import QPController
from compatible_clf_cbf.dynamic_simulation import SimulateDynamics
from compatible_clf_cbf.graphical_simulation import GraphicalSimulation
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier

try:
    # Simulation parameters
    dt = .005
    sim_freq = 1/dt
    T = 20

    # Define 2D plant and initial state
    f = ['0','0']
    g1 = ['1','0']
    g2 = ['0','1']
    state_string = 'x1, x2'
    control_string = 'u1, u2'
    plant = AffineSystem(state_string, control_string, f, g1, g2)

    # Define initial state for plant simulation
    x_init = 0.1
    y_init = 5
    initial_state = np.array([x_init,y_init])

    # Create CLF
    lambda_x, lambda_y = 2.0, 1.0
    Hv = np.array([ [ lambda_x , 0.0 ],
                    [ 0.0 , lambda_y ] ])
    x0 = np.array([0,0])
    clf = QuadraticLyapunov(state_string, Hv, x0)

    # Create CBF
    xaxis_length, yaxis_length = 2.0, 1.0
    lambda1, lambda2 = 1/xaxis_length**2, 1/yaxis_length**2
    Hh = np.array([ [ lambda1 , 0.0 ],
                    [ 0.0 , lambda2 ] ])
    p0 = np.array([0,3])
    cbf = QuadraticBarrier(state_string, Hh, p0)
    
    # lambda1, lambda2 = 1, 1
    # Kappa = np.array([[lambda1, 0],[0, lambda2]])

    # Create QP controller
    ref = np.array([0,0])
    qp_controller = QPController(plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 10.0)

    # Initialize simulation object
    dynamicSimulation = SimulateDynamics(plant, initial_state)
    graphicalSimulation = GraphicalSimulation(ref, clf, cbf)

    # Main loop
    rate = rospy.Rate(sim_freq)
    while not rospy.is_shutdown():

        # Get simulation state
        state = dynamicSimulation.state()

        # Control
        # ref = graphicalSimulation.get_reference()
        # error = state - ref
        # control = - Kappa.dot(error)
        control = qp_controller.compute_control(state)

        # Send actuation commands 
        dynamicSimulation.send_control_inputs(control, dt)
        
        # if np.linalg.norm(error) < 0.1:
        #     ref = np.random.randint(low = -10, high = 10, size = 2)
        #     graphicalSimulation.draw_reference(ref)

        # Draw graphical simulation
        graphicalSimulation.draw_trajectory(state)
        graphicalSimulation.draw_reference(ref)
        graphicalSimulation.draw_clf(state)
        graphicalSimulation.draw_cbf()

        rate.sleep()

except rospy.ROSInterruptException:
    pass