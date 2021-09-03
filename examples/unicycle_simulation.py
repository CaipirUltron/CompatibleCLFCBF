import math
import numpy as np
from compatible_clf_cbf.dynamic_systems import Unicycle

# Mobile robot with unicycle dynamics: state = [x, y, phi], control = [v, omega]
initial_state = [0.0, 0.0, math.pi/2]
initial_control = [0.0, 0.0]
robot = Unicycle(initial_state, initial_control)

dt = 0.01
sim_time = 10
for t in np.linspace(0.0, sim_time, num=int(sim_time/dt)):

    # Control
    u_control = [ 1.0, 0.0 ]
    
    # Send actuation commands
    robot.set_control(u_control) 
    robot.actuate(dt)

    state = robot.get_state()
    print( "Robot state (t = " + str(t) + ") is x = " + str(state[0]) + ", y = " + str(state[1]) + ", phi = " + str(state[2]) + ") "   )