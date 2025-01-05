'''
Compatibilizes CLF with CBF and plant for a given KernelTriplet.
The results are stored in a json file.
'''
import sys
import json
import importlib
import numpy as np

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

if sim.kerneltriplet.is_compatible():
    print("Given CLF is already compatible with CBF and plant.")
else: 
    print("Given CLF is not compatible with CBF and plant.")

    Ninit, _ = sim.kerneltriplet.get_N(sim.Pquadratic)
    compatibility_result = sim.kerneltriplet.compatibilize(Ninit, sim.clf_center, verbose=True, animate=True)

    try:
        with open("logs/"+simulation_file+"_comp.json", "w") as file:
            json.dump(compatibility_result, file, indent=4)
            print("Compatibilization file saved successfully.")
    except IOError:
        print("Error saving compatibilization file.")