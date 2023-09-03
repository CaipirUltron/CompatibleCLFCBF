import sys
import json
import importlib
import numpy as np
import matplotlib.pyplot as plt
from graphics import Plot2DSimulation

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

sim.plot_config["figsize"] = (10,5)
sim.plot_config["gridspec"] = (1,2,1)
sim.plot_config["widthratios"] = [1,1]
sim.plot_config["heightratios"] = [1]
plotSim = Plot2DSimulation( logs, sim.plant, sim.clf, sim.cbfs, plot_config = sim.plot_config )

input_ax = plotSim.main_ax
output_ax = plotSim.fig.add_subplot(122)
canvas = plotSim.fig.canvas

background = canvas.copy_from_bbox(plotSim.fig.bbox)
canvas.blit(plotSim.fig.bbox)

plotSim.plot_frame(5.0)

N_list = sim.kernel.get_N_matrices()
n = sim.kernel._dim
p = sim.kernel.kernel_dim

c = 1
def L(l, z):
    """Returns the z dependent pencil"""
    return 0.5 * c * z.T @ sim.P @ z * sim.P  - l * sim.Q

def kappa_part(z, kappa):
    M = sim.F 
    for k in range(len(N_list)):
        M += kappa[k] * N_list[k]
    return M @ z

def q_function(l, z, kappa):
    v = kappa_part(z, kappa)
    y = np.linalg.inv( L(l, z) ) @ v
    return ( y.T @ sim.Q @ y )

def projection(z):
    a = np.zeros(p-n)
    for k in range(p-n):
        a[k] = z.T @ ( N_list[k] - N_list[k].T ) @ z
    if np.linalg.norm(a) > 0.00001:
        P = np.eye(p-n) - np.outer(a,a)/np.linalg.norm(a)**2
    else: P = np.eye(p-n)
    return P

def compute_q(l_list, x, w):
    z = sim.kernel.function(x)
    kappa = projection(z) @ w
    q_list = np.zeros(len(l_list))
    for k in range(len(q_list)):
        try:
            q_list[k] = q_function(l_list[k], z, kappa)
        except np.linalg.LinAlgError as error:
            q_list[k] = np.inf
            print(error)
    # q_list = l_list**2
    output_ax.plot( l_list, q_list, 'b-', linewidth=0.5)

l_range = [0, 100]
q_range = [0, 200]
output_ax.set_xlim(l_range)
output_ax.set_ylim(q_range)
l = np.arange(l_range[0], l_range[1], 0.5, dtype='float64')

w = np.random.rand(p-n)

# def on_draw(event):
#     """Callback for draws."""
#     background = canvas.copy_from_bbox(output_ax.bbox)
#     canvas.blit(output_ax.bbox)

#     yield background

def on_mouse_move(event):
    """Callback for mouse movements."""

    canvas.restore_region(background)    
    output_ax.cla()
    if event.inaxes == input_ax:
        x = np.array([ event.x, event.y ])
        
        compute_q(l, x, w)
        output_ax.set_xlim(l_range)
        output_ax.set_ylim(q_range)
        canvas.draw()
    
# input_canvas.mpl_connect('button_press_event', on_button_press)
# input_canvas.mpl_connect('key_press_event', on_key_press)
# input_canvas.mpl_connect('button_release_event', on_button_release)
# canvas.mpl_connect('draw_event', on_draw)
canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.show()