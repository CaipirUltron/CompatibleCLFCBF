import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functions import canonical2D, QuadraticLyapunov, Gaussian

clf_lambda_x, clf_lambda_y, clf_angle = 3.0, 1.0, math.radians(0.0)
clf_params = {
    "Hv": canonical2D([ clf_lambda_x , clf_lambda_y ], clf_angle),
    "x0": [ 0.0, 0.0 ] }
initial_state = [2.0, 2.5]
clf_quadratic = QuadraticLyapunov(*initial_state, hessian = clf_params["Hv"], critical = clf_params["x0"])
clf_bump = Gaussian(*initial_state, constant=1.0, mean=[1.0, 0.0], shape=np.eye(2))

fig, ax = plt.subplots()

artists = []
max_range = 10
for k in range(max_range):
    cs = clf_bump.contour_plot(ax, levels=[max_range - k], colors=['blue'], min=-6, max=6, resolution=0.2)
    artists.append(cs.collections)

animation.ArtistAnimation(fig, artists, interval=50, blit=True, repeat_delay=1000)

plt.show()