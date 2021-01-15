import numpy as np
from dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier
from dynamic_simulation import SimulateDynamics

# Simulation parameters
dt = .01
T = 20

# Plant
f = ['0']
g = ['1']
plant = AffineSystem('x1','u1',f,g)
initial_state = np.array([0])
simulation = SimulateDynamics(plant, initial_state)

# Controller
k = 1
ref = 2

# Simulation loop
t = 0
nb_steps = int(T/dt)
for _ in range(0, nb_steps):

    # Simulation time
    t += dt

    # State
    state = simulation.state()
    print("State: ", state)

    # Control
    control = - k * np.array([ state[0] - ref ])

    # Actuate
    simulation.send_control_inputs(control, dt)