import sys
import importlib
import matplotlib.pyplot as plt

graphics = importlib.import_module("graphics.gen_graphics", package=None)

time = 0.0
if len(sys.argv) > 3:
    time = float(sys.argv[3])

graphics.plotSim.plot_frame(time)

plt.show()