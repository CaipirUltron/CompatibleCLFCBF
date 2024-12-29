import importlib
import platform, os, sys, json, time

from controllers import CompatibleQP

''' ------------------------- Load simulation example ------------------------------ '''
sim_config = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+sim_config, package=None)

''' ---------------------------- Load controller ---------------------------------- '''
control_mode = 'nominal'
# control_mode = 'compatible'
# control_mode = 'no_control'

if len(sys.argv) > 2:
    control_mode = float(sys.argv[2])

if control_mode == 'nominal':
    control_opts = {"compatibilization": False, 
                    "active": True}

if control_mode == 'nominal':
    control_opts = {"compatibilization": False, 
                    "active": True}
    
if control_mode == 'no_control':
    control_opts = {"compatibilization": False, 
                    "active": False}
    
sample_time = 2e-2
controller = CompatibleQP(sim.plant, sim.clf, sim.cbfs, alpha = [1.0, 10.0], beta = 1.0, p = [1.0, 1.0], 
                          dt = sample_time,
                          **control_opts,
                          verbose=True)

''' ----------------------------- Simulation loop --------------------------------- '''
T = 10
if len(sys.argv) > 3:
    T = float(sys.argv[3])

logs = { "sample_time": sample_time }
num_steps = int(sim.T/sample_time)
time_list = []

print('Running simulation...')
time.sleep(1.5)

for step in range(0, num_steps):

    # Simulation time
    t = step*sample_time
    if platform.system().lower() != 'windows':
        os.system('var=$(tput lines) && line=$((var-2)) && tput cup $line 0 && tput ed')           # clears just the last line of the terminal
    print("Simulating instant t = " + str(float(f'{t:.3f}')) + " s")
    time_list.append( t )

    # Control
    u_control = controller.get_control()
    upi_control = controller.get_clf_control()

    # Send actuation commands
    controller.update_clf_dynamics(upi_control)
    sim.plant.set_control(u_control) 
    sim.plant.actuate(sample_time)

# Collect simulation logs and save in .json file ------------------------------------
sim.logs["time"] = time_list
sim.logs["state"] = sim.plant.state_log
sim.logs["control"] = sim.plant.control_log
sim.logs["clf_log"] = controller.clf.dynamics.state_log
sim.logs["equilibria"] = controller.equilibrium_points
sim.logs["tracking"] = None

if hasattr(sim, 'path'):
    sim.logs["gamma_log"] = sim.path.logs["gamma"]

with open("logs/"+sim_config+"-"+control_mode+".json", "w") as file:
    print("Saving simulation data...")
    json.dump(sim.logs, file, indent=4)