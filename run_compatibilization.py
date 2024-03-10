import sys
import json
import time
import importlib

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

start_time = time.time()

Pcompatible = sim.kerneltriplet.compatibilize(verbose=True)

end_time = time.time()
execution_time = end_time - start_time
print("Compatibilization took ", execution_time, "s")

with open("logs/"+simulation_file+"_compatibility.json", "w") as file:
    print("Saving compatibilization matrix...")
    compatibility_dict = {"Pcompatible": Pcompatible.tolist()}
    json.dump(compatibility_dict, file, indent=4)