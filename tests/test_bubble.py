import itertools
import numpy as np
import contourpy as ctp
import matplotlib.pyplot as plt
from copy import copy

from matplotlib.backend_bases import MouseButton

def inside(num, interval):
    return num >= interval[0] and num <= interval[1]

def interval_distance(inter1, inter2):
    '''
    Computes distance between two intervals: positive if intervals do not overlap, 0 otherwise
    '''
    r1, r2 = inter1["limits"], inter2["limits"]
    x, y = sorted((r1, r2))

    if x[0] <= x[1] < y[0] and all( y[0] <= y[1] for y in (r1,r2)):
        return y[0] - x[1]
    return 0

def continuous_composition(k, *args):
    ''' Continuously compose any number of functions '''    
    return -(1/k)*np.log(sum([ np.exp( -k * arg ) for arg in args ]))

class IntervalEditor:
    """
    A simple interactive editor for Bubble functions
    """
    def __init__(self, ax, *intervals):

        self.plotconfigs = {"xlim": (-10, 10),
                            "ylim": (-10, 10),
                            "res": 0.05 }

        self.composite_k = 10
        self.yaxis = 1.0
        self.percent_to_bounds = 10

        self.intervals = intervals
        self.barrier_funs = [ self._quadratic_barrier(interval) for interval in self.intervals ]

        # self.graph = spline_graph
        self.ax = ax
        self.canvas = self.ax.figure.canvas

        self.default_line, = self.ax.plot( [], [], color='b', linestyle='solid', alpha=0.9 )
        self.contour_lines = []

        # self.path = spline_path
        # x, y = self.path.points[:,0], self.path.points[:,1]
        # self.line, = self.ax.plot(x, y, marker='o', markerfacecolor='r', animated=True)

        self.selected_index = -1
        self.new_interval = None

        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        # self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self._generate_contour()
        self.canvas.draw()

    def _quadratic_barrier(self, interval):
        ''' Defines quadratic barrier function for interval '''

        min_bound, max_bound = interval[0], interval[1]
        length = np.abs(max_bound - min_bound)
        center = (min_bound + max_bound)/2

        a = length/2
        b = self.yaxis

        return lambda pt: (1/(a**2))*( pt[0] - center )**2 + (1/(b**2))*( pt[1] )**2 - 1

    def _barrier(self, pt):
        k = self.composite_k
        barrier_vals = [ barrier_fun(pt) for barrier_fun in self.barrier_funs ]
        return continuous_composition(k, *barrier_vals)

    def update_interval(self, index, new_interval):

        self.barrier_funs[index] = self._quadratic_barrier(new_interval)
        self._generate_contour()

    def _generate_contour(self):
        '''
        Create contour generator object for the given function.
        Parameters: limits (2x2 array) - min/max limits for x,y coords
                    spacing - grid spacing for contour generation
        '''    
        x_min, x_max = self.plotconfigs["xlim"]
        y_min, y_max = self.plotconfigs["ylim"]
        res = self.plotconfigs["res"]

        x = np.arange(x_min, x_max, res)
        y = np.arange(y_min, y_max, res)
        xg, yg = np.meshgrid(x,y)
        
        fvalues = np.zeros(xg.shape)
        for i,j in itertools.product(range(xg.shape[0]), range(xg.shape[1])):
            pt = np.array([xg[i,j], yg[i,j]])
            fvalues[i,j] = self._barrier(pt)
        
        self.contour = ctp.contour_generator(x=xg, y=yg, z=fvalues )

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        hvalues_at_pt = self._barrier( (event.xdata, event.ydata) )

        xt, yt = self.path.points[:,0], self.path.points[:,1]
        d = np.sqrt((xt - event.xdata)**2 + (yt - event.ydata)**2)
        ind = d.argmin()

        return ind if d[ind] < self.epsilon else None

    def draw_contour(self):
        level = self.contour.lines(0.0)
        for k, segment in enumerate(level):

            if k >= len(self.contour_lines):
                self.contour_lines.append( copy(self.default_line) )

            self.contour_lines[k].set_data( segment[:,0], segment[:,1] )
            self.ax.draw_artist(self.contour_lines[k])

        for i in range(k+1, len(self.contour_lines)):
            self.contour_lines[i].set_data([],[])

    def on_draw(self, event):
        ''' Callback for draws '''
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.draw_contour()
        self.canvas.blit(self.ax.bbox)

    def on_button_press(self, event):
        ''' Callback for mouse button presses '''

        if ( event.inaxes is None or 
             event.button != MouseButton.LEFT or 
             self.new_interval is None or 
             self.new_interval is self.intervals[self.selected_index]):
            return

        print("button pressed")

        self.update_interval(self.selected_index, self.new_interval)
        self.draw_contour()

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if (event.button != MouseButton.LEFT):
            return
        
        self.selected_index = -1
        self.new_interval = None

    def on_mouse_move(self, event):
        """Callback for mouse movements."""

        if (event.inaxes is None
            or event.button != MouseButton.LEFT):
            return

        print("pressed and mouse is moving")

        pos_x, pos_y = event.xdata, event.ydata

        if pos_x:
            for k, interval in enumerate(self.intervals):
                if inside(pos_x, interval):
                    self.selected_index = max(k, self.selected_index)

            # Selected interval
            if self.selected_index >= 0:

                selected = self.intervals[self.selected_index]
                self.new_interval = selected

                length = np.abs(selected[1]-selected[0])
                percent = 100*((pos_x-interval[0])/length)

                if percent <= self.percent_to_bounds:               # Change left bound
                    self.new_interval = [ pos_x, selected[1] ]
                elif percent >= 100-self.percent_to_bounds:         # Change right bound
                    self.new_interval = [ selected[0], pos_x ]
                else:                                               # Change position
                    self.new_interval = [ pos_x - length/2, pos_x + length/2 ]

                self.update_interval(self.selected_index, self.new_interval)

        # if (self._ind is None
        #         or event.inaxes is None
        #         or event.button != MouseButton.LEFT):
        #     return

        # pts = self.path.points
        # pts[self._ind,0], pts[self._ind,1] = event.xdata, event.ydata

        # self.path.set_points(pts)
        # x, y = pts[:,0], pts[:,1]
        # self.line.set_data(x, y)

        self.canvas.restore_region(self.background)
        self.draw_contour()
        self.canvas.blit(self.ax.bbox)
        self.canvas.draw()

''' --------------------------------- Animation ------------------------------------- '''
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8.0, 5.0), layout="constrained")
fig.suptitle('Test Bubble Function')

ax.set_xlim( (-1, 10) )
ax.set_ylim( (-2, 2) )
ax.set_aspect('equal', adjustable='box')

intervals = [ [0,1], [1,3], [3,6] ]
editor = IntervalEditor(ax, *intervals)
plt.show()