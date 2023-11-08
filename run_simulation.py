import os
import sys
import json
import importlib

simulation_config = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_config, package=None)

# Simulation loop -------------------------------------------------------------------
T = 10
num_steps = int(T/sim.sample_time)
time = []
print('Running simulation...')
for step in range(0, num_steps):

    # Simulation time
    t = step*sim.sample_time
    os.system('clear')
    print("Simulating instant t = " + str(float(f'{t:.3f}')) + " s")
    time.append( t )

    # Control
    u_control = sim.controller.get_control()
    upi_control = sim.controller.get_clf_control()

    # Send actuation commands
    sim.controller.update_clf_dynamics(upi_control)
    sim.plant.set_control(u_control) 
    sim.plant.actuate(sim.sample_time)

# Collect simulation logs and save in .json file ------------------------------------
sim.logs["time"] = time
sim.logs["state"] = sim.plant.state_log
sim.logs["control"] = sim.plant.control_log
sim.logs["clf_log"] = sim.controller.clf.dynamics.state_log
sim.logs["equilibria"] = sim.controller.equilibrium_points.tolist()

if hasattr(sim, 'path'):
    sim.logs["gamma_log"] = sim.path.logs["gamma"]

with open("logs/"+simulation_config+".json", "w") as file:
    print("Saving simulation data...")
    json.dump(sim.logs, file, indent=4)
