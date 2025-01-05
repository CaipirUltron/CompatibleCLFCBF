import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from numpy.polynomial import Polynomial as Poly
# from controllers.compatibility import root_barrier

class Barrier:
    """
    A simple interactive editor for BSpline.

    Press 't' to toggle vertex markers on and off.  When vertex markers are on,
    they can be dragged with the mouse.
    """
    def __init__(self, ax):

        self.ax = ax
        canvas = self.ax.figure.canvas

        self.intervals = [ (-np.inf, 10), (20, 40) ]

        min_l, max_l = -100, +100
        for interval in self.intervals:

            min_rect_bound = max(interval[0], min_l)
            max_rect_bound = min(interval[1], max_l)
            length = np.abs(max_rect_bound - min_rect_bound)

            rect = patches.Rectangle((min_rect_bound, -0.5), length, 1, facecolor='r', alpha=.6)

            print(rect)
            ax.add_patch(rect)

        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

    def on_mouse_move(self, event):
        """Callback for mouse movements."""

        if (event.inaxes is None):
            return

        os.system('cls' if os.name == 'nt' else 'clear')

        for k, interval in enumerate(self.intervals):
            root = complex(event.xdata, event.ydata)
            barrier = root_barrier( root, interval )
            print(f"{k+1} root barrier = {barrier}")

        self.canvas.blit(self.ax.bbox)
        self.canvas.draw()

def root_barrier(interval, t):

    a, b = interval[0], interval[1]
    c = (1-t)*a + t*b
    length = b - a
    
    s1 = 1/(c-a)**2
    left_quad = s1*Poly([-a, 1])*Poly([-2*c+a, 1])

    s2 = 1/(b-c)**2
    right_quad = s2*Poly([-b, 1])*Poly([-2*c+b, 1])

    def fun(x):
        if x < c: return left_quad(x)
        else: return right_quad(x)
        
    return fun

def root_barriers(intervals, complex_conjs):

    for interval in intervals:
        pass
    
intervals = [ (1,5), (6,8) ]

size = 6.0
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.4*size, size), layout="constrained")
fig.suptitle('Test new barrier')
ax.set_aspect('auto')

min_l, max_l = 0, +20
l_arr = np.arange(min_l, max_l, 0.1)

for k, interval in enumerate(intervals):

    min_rect_bound = max(interval[0], min_l)
    max_rect_bound = min(interval[1], max_l)
    length = np.abs(max_rect_bound - min_rect_bound)

    rect = patches.Rectangle((min_rect_bound, -0.5), length, 1, facecolor='r', alpha=.4)
    ax.add_patch(rect)

    if k == 0: t = 0.9
    else:      t = 0.1

    barrier = root_barrier(interval, t=t)
    b_arr = [ barrier(l) for l in l_arr ]
    ax.plot(l_arr, b_arr, 'b--', label=f'{k+1}')

ax.set_xlim(min_l,max_l)
ax.set_ylim(-10,10)
ax.grid()

plt.show()