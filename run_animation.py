import sys
import importlib
import matplotlib.pyplot as plt

graphics = importlib.import_module("graphics.gen_graphics", package=None)

initial_time = 0.0
if len(sys.argv) > 3:
    initial_time = float(sys.argv[3])

animation = graphics.plotSim.animation(graphics.fig, initial_time)
# graphics.plotSim.animation.save(simulation_file + ".mp4", writer=anim.FFMpegWriter(fps=30, codec='h264'), dpi=100)

plt.show()