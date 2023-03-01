import sys
import json
import importlib

# simulation_file = "integrator_nominalQP"
simulation_config = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_config, package=None)

# Simulation loop -------------------------------------------------------------------
T = 20
num_steps = int(T/sim.sample_time)
time = [0]
print('Running simulation...')
for step in range(1, num_steps+1):

    # Simulation time
    time.append( step*sim.sample_time )

    # Control
    u_control = sim.controller.get_control()
    upi_control = sim.controller.get_clf_control()

    # Send actuation commands
    sim.controller.update_clf_dynamics(upi_control)
    sim.plant.set_control(u_control) 
    sim.plant.actuate(sim.sample_time)

# Collect simulation logs and save in .json file ------------------------------------
logs = {
    "time": time,
    "sample_time": sim.sample_time,
    "state": sim.plant.state_log,
    "control": sim.plant.control_log,
    "clf_log": sim.controller.clf.dynamics.state_log,
    "equilibria": sim.controller.equilibrium_points.tolist(),
    "mode": sim.controller.mode_log
}

with open(simulation_config+".json", "w") as file:
    print("Saving simulation data...")
    json.dump(logs, file, indent=4)