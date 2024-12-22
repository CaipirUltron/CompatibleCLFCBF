import importlib
import numpy as np
import platform, os, sys, json, time

simulation_config = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_config, package=None)

# Simulation loop -------------------------------------------------------------------
num_steps = int(sim.T/sim.sample_time)
time_list = []

print('Running simulation...')
time.sleep(1.5)

for step in range(0, num_steps):

    # Simulation time
    t = step*sim.sample_time
    if platform.system().lower() != 'windows':
        os.system('var=$(tput lines) && line=$((var-2)) && tput cup $line 0 && tput ed')           # clears just the last line of the terminal
    print("Simulating instant t = " + str(float(f'{t:.3f}')) + " s")
    time_list.append( t )

    # Control
    u_control = sim.controller.get_control()
    upi_control = sim.controller.get_clf_control()

    # Send actuation commands
    sim.controller.update_clf_dynamics(upi_control)
    sim.plant.set_control(u_control) 
    sim.plant.actuate(sim.sample_time)

# Collect simulation logs and save in .json file ------------------------------------
sim.logs["time"] = time_list
sim.logs["state"] = sim.plant.state_log
sim.logs["control"] = sim.plant.control_log
sim.logs["clf_log"] = sim.controller.clf.dynamics.state_log
sim.logs["equilibria"] = sim.controller.equilibrium_points
sim.logs["tracking"] = None

if hasattr(sim, 'path'):
    sim.logs["gamma_log"] = sim.path.logs["gamma"]

with open("logs/"+simulation_config+".json", "w") as file:
    print("Saving simulation data...")
    json.dump(sim.logs, file, indent=4)