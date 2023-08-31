import sys
import json
import importlib
import numpy as np
import matplotlib.pyplot as plt
from graphics import Plot2DSimulation
from controllers import compute_equilibria_algorithm7, get_boundary_points

# Load simulation file
simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

try:
    with open("logs/"+simulation_file + ".json") as file:
        print("Loading graphical simulation with "+simulation_file + ".json")
        logs = json.load(file)
except IOError:
    print("Couldn't locate " + simulation_file + ".json")

print('Animating simulation...')

# -----------------------------------Plots starting of simulation ------------------------------------------
plotSim = Plot2DSimulation( logs, sim.plant, sim.clf, sim.cbfs, plot_config = sim.plot_config )
plotSim.plot_frame(5.0)

# boundary_pt = find_nearest_boundary(sim.cbf, [-7.0, 0.0])
# print("Boundary initializer = " + str(boundary_pt))
# plotSim.main_ax.plot( boundary_pt[0], boundary_pt[1], 'go')

# equilibrium = compute_equilibria_algorithm7(sim.plant, sim.clf, sim.cbf, boundary_pt, c = 1)
# if np.any(equilibrium) != None:
#     print("Found equilibrium points at :\n" + str(equilibrium))
#     plotSim.main_ax.plot( equilibrium[0], equilibrium[1], 'ro')

N = 1
# initial_guesses = 4*2*(np.random.rand(N,2)-0.5)
initial_guesses = np.array([ sim.initial_state ])
boundary_pts = get_boundary_points(sim.cbf, initial_guesses)
plotSim.main_ax.plot( boundary_pts[:,0], boundary_pts[:,1], 'g*' )

solutions = compute_equilibria_algorithm7(sim.plant, sim.clf, sim.cbf, boundary_pts, c = 1)

pts = np.array( solutions["points"] )
initial_pts = initial_guesses[ solutions["indexes"], : ]

num_convergences = np.shape(pts)[0]

print("From " + str(N) + " points, algorithm converged " + str(num_convergences) + " times.")
print("Algorithm efficiency = " + str(num_convergences/N*100) + "%" )

plotSim.main_ax.plot( initial_pts[:,0], initial_pts[:,1], 'b*' )
plotSim.main_ax.plot( pts[:,0], pts[:,1], 'r*' )

plt.show()