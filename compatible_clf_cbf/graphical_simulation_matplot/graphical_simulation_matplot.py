import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from compatible_clf_cbf.dynamic_systems import Quadratic

class SimulationMatplot():

    def __init__(self, axes_lim, numpoints, logs, clf, cbf, draw_level = False, draw_gradient = False):

        self.draw_level = draw_level
        self.draw_gradient = draw_gradient

        # Initialize plot objects
        self.fig = plt.gcf()
        self.ax = plt.axes(xlim=axes_lim[0:2], ylim=axes_lim[2:4])
        self.ax.set_title('CLF-CBF QP-based Control')

        # Get logs
        self.time = logs["time"]
        self.state_log = logs["stateLog"]
        self.clf_log = logs["clfLog"]
        self.cbf_log = logs["cbfLog"]
        # self.normal_vec = logs["normal"]
        self.num_steps = len(self.state_log[0])

        # Get point resolution for graphical objects
        self.numpoints = numpoints

        self.clf, self.cbf = clf, cbf

        # Initalize graphical objects
        self.time_text = self.ax.text(axes_lim[1]-2.5, axes_lim[3]-1, str("Time = "))
        self.trajectory, = self.ax.plot([],[],lw=2)
        self.clf_level_set1, = self.ax.plot([],[],'b',lw=1)
        self.clf_level_set2, = self.ax.plot([],[],'b',lw=1)
        self.cbf_level_set1, = self.ax.plot([],[],'g',lw=1)
        self.cbf_level_set2, = self.ax.plot([],[],'g',lw=1)

        clf_crit = self.clf.get_critical()
        cbf_crit = self.cbf.get_critical()

        self.clf_crit, = self.ax.plot(clf_crit[0], clf_crit[1], 'bo--', linewidth=1, markersize=2)
        self.cbf_crit, = self.ax.plot(cbf_crit[0], cbf_crit[1], 'go--', linewidth=1, markersize=2)

        # self.normal_arrow = self.ax.quiver([0.0],[0.0],[0.1],[0.1],pivot='mid',color='r')
        # self.clf_arrows = self.ax.quiver([0.0],[0.0],[0.1],[0.1],pivot='mid',color='b')
        # self.cbf_arrows = self.ax.quiver([0.0],[0.0],[0.1],[0.1],pivot='mid',color='g')

        self.animation = None

    def init(self):

        self.time_text.text = str("Time = ")
        self.trajectory.set_data([],[])

        self.clf_level_set1.set_data([],[])
        self.clf_level_set2.set_data([],[])

        self.cbf_level_set1.set_data([],[])
        self.cbf_level_set2.set_data([],[])

        return self.trajectory, self.clf_level_set1, self.clf_level_set2, self.cbf_level_set1, self.cbf_level_set2, self.clf_crit, self.cbf_crit

    def update(self, i):

        xdata, ydata = self.state_log[0][0:i], self.state_log[1][0:i]
        self.trajectory.set_data(xdata, ydata)

        current_time = np.around(self.time[i], decimals = 2)
        current_state = [self.state_log[0][i], self.state_log[1][i]]
        current_piv_state = [self.clf_log[0][i], self.clf_log[1][i], self.clf_log[2][i]]

        self.time_text.set_text("Time = " + str(current_time) + "s")

        Hv = Quadratic.vector2sym(current_piv_state)
        self.clf.set_param(hessian=Hv)
        if self.draw_level:
            V = self.clf.evaluate(current_state)
            xclf, yclf, uclf, vclf = self.clf.superlevel(V, self.numpoints)
            self.clf_level_set1.set_data(xclf[0], yclf[0])
            self.clf_level_set2.set_data(xclf[1], yclf[1])

        current_pih_state = [self.cbf_log[0][i], self.cbf_log[1][i], self.cbf_log[2][i]]
        Hh = Quadratic.vector2sym(current_pih_state)
        self.cbf.set_param(hessian=Hh)
        
        xcbf, ycbf, ucbf, vcbf = self.cbf.superlevel(0.0, self.numpoints)
        self.cbf_level_set1.set_data(xcbf[0], ycbf[0])
        self.cbf_level_set2.set_data(xcbf[1], ycbf[1])

        if self.draw_gradient:
        # self.clf_arrows = self.ax.quiver(xclf, yclf, uclf, vclf, pivot='tail', color='b', scale=50.0, headlength=0.5, headwidth=1.0)
            self.cbf_arrows = self.ax.quiver(xcbf, ycbf, ucbf, vcbf, pivot='tail', color='g', scale=50.0, headlength=0.5, headwidth=1.0)

        # if self.draw_arrows:
        #     self.normal_arrow = self.ax.quiver(xdata, ydata, self.normal_vec[0], self.normal_vec[1], pivot='tail', color='r', scale=50.0, headlength=0.5, headwidth=1.0)

        return self.time_text, self.trajectory, self.clf_level_set1, self.clf_level_set2, self.cbf_level_set1, self.cbf_level_set2, self.clf_crit, self.cbf_crit

    def animate(self):
        self.animation = anim.FuncAnimation(self.fig, func=self.update, init_func=self.init, frames=self.num_steps, interval=20, repeat=False, blit=True)