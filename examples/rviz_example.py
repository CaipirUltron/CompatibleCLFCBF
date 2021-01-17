import rospy
import numpy as np

from compatible_clf_cbf.dynamic_simulation import SimulateDynamics
from compatible_clf_cbf.graphical_simulation import GraphicalSimulation
from compatible_clf_cbf.dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier

try:
    # Simulation parameters
    dt = .01
    sim_freq = 1/dt
    T = 20

    # Define plant and initial state
    f = ['0','0']
    g1 = ['1','0']
    g2 = ['0','1']
    state_string = 'x1, x2'
    control_string = 'u1, u2'
    plant = AffineSystem(state_string, control_string, f, g1, g2)
    initial_state = np.array([0,0])

    # Create CLF and CBF
    Hv = np.array([ [ 1.0 , 0.0 ],
                    [ 0.0 , 1.0 ] ])
    x0 = np.array([0,0])

    Hh = np.array([ [ 1.0 , 0.0 ],
                    [ 0.0 , 1.0 ] ])
    p0 = np.array([0,1])

    clf = QuadraticLyapunov(state_string, Hv, x0)
    cbf = QuadraticBarrier(state_string, Hh, p0)
    
    # Provisional linear controller (to be removed)
    lambda1, lambda2 = 1, 1
    Kappa = np.array([[lambda1, 0],[0, lambda2]])
    ref = np.array([4,4])

    # Initialize simulation object
    dynamicSimulation = SimulateDynamics(plant, initial_state)
    graphicalSimulation = GraphicalSimulation(ref)

    # Main ROS loop
    rate = rospy.Rate(sim_freq)
    while not rospy.is_shutdown():

        # Get simulation state
        state = dynamicSimulation.state()

        # Control
        # ref = graphicalSimulation.get_reference()
        error = state - ref
        control = - Kappa.dot(error)

        # Send actuation commands 
        dynamicSimulation.send_control_inputs(control, dt)
        
        if np.linalg.norm(error) < 0.1:
            ref = np.random.randint(low = -10, high = 10, size = 2)
            graphicalSimulation.set_reference(ref)

        # Draw graphical simulation
        graphicalSimulation.draw_trajectory(state)

        rate.sleep()

except rospy.ROSInterruptException:
    pass