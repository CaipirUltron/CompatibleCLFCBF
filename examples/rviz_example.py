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
    plant = AffineSystem('x1, x2','u1, u2',f,g1, g2)
    initial_state = np.array([0,0])

    # Create CLF and CBF
    # ... not implemented
    lambda1, lambda2 = 1, 1
    Kappa = np.array([[lambda1, 0],[0, lambda2]])

    # Initialize simulation object
    simulation = SimulateDynamics(plant, initial_state)
    graphicalSimulation = GraphicalSimulation()

    # Initialize node
    rospy.init_node('graphics_broadcaster', anonymous = True)
    rospy.loginfo("Starting graphical simulation...")
    
    # Main ROS loop
    rate = rospy.Rate(sim_freq)
    while not rospy.is_shutdown():

        # Get simulation state
        state = simulation.state()

        # Control
        # control = np.random.normal(size=2)
        ref = np.array([5,-2])
        error = state - ref
        control = - Kappa.dot(error)

        # Send actuation commands 
        simulation.send_control_inputs(control, dt)
        
        # Draw graphical simulation
        graphicalSimulation.draw_trajectory(state)

        rate.sleep()

except rospy.ROSInterruptException:
    pass