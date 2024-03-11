'''
Compatibilizes CLF with CBF and plant for a given KernelTriplet.
The results are stored in a json file.
'''
import sys
import json
import importlib

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

if sim.kerneltriplet.is_compatible(): 
    print("Given CLF is already compatible with CBF and plant.")
else: 
    print("Given CLF is not compatible with CBF and plant.")

    compatibility_result = sim.kerneltriplet.compatibilize(verbose=True)

    try:
        with open("logs/"+simulation_file+"_comp.json", "w") as file:
            json.dump(compatibility_result, file, indent=4)
            print("Compatibilization file saved successfully.")
    except IOError:
        print("Error saving compatibilization file.")