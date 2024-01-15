import sys, json
import importlib
from controllers.equilibrium_algorithms import closest_compatible

simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

try:
    with open("logs/"+simulation_file + ".json") as file:
        print("Loading graphical simulation with "+simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate " + simulation_file + ".json")

P = closest_compatible(sim.plant, sim.clf, sim.cbf, logs["equilibria"], slack_gain=sim.p, clf_gain=sim.alpha)
print(P)