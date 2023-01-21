import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


class SimulationMatplot():
    '''
    Class for Matplot-based simulation.
    '''
    def __init__(self, axes_lim, numpoints, logs, clf, cbfs, draw_level = False):

        self.fps = 50
        self.draw_level = draw_level

        # Initialize plot objects
        self.fig = plt.gcf()
        self.x_lim = axes_lim[0:2]
        self.y_lim = axes_lim[2:4]
        self.ax = plt.axes(xlim=self.x_lim, ylim=self.y_lim)
        self.ax.set_title('CLF-CBF QP-based Control')

        # Get logs
        self.time = logs["time"]
        self.state_log = logs["state"]
        if "clf" in logs.keys():
            self.clf_log = logs["clf"]
        # if "cbfLog" in logs.keys():
        #     self.cbf_log = logs["cbfLog"]
        self.mode_log = logs["mode"]
        self.equilibria = logs["equilibria"]
        self.num_steps = len(self.state_log[0])
        self.anim_step = (self.num_steps/self.time[-1])/self.fps
        self.current_step = 0
        self.runs = False

        # Get point resolution for graphical objects
        self.numpoints = numpoints

        self.clf, self.cbfs = clf, cbfs
        self.num_cbfs = len(self.cbfs)

        # Initalize graphical objects
        self.time_text = self.ax.text(axes_lim[1]-2.5, axes_lim[3]-11.5, str("Time = "))
        self.mode_text = self.ax.text(axes_lim[0]+0.5, axes_lim[3]-1, "", fontweight="bold")

        self.trajectory, = self.ax.plot([],[],lw=2)
        self.init_state, = self.ax.plot([],[],'bo',lw=2)
        self.equilibria_plot, = self.ax.plot([],[], marker='o', mfc='none', lw=2, color=[1.,0.,0.], linestyle="None")

        self.clf_contours = self.clf.contour_plot(self.ax, levels=[0.0], colors=['blue'], min=self.x_lim[0], max=self.x_lim[1], resolution=0.5)
        self.cbf_contours = []
        # self.cbf_contours = self.cbf.contour_plot(self.ax, levels=[0.0], colors=['green'], min=self.x_lim[0], max=self.x_lim[1], resolution=0.1)

        self.animation = None

    def init(self):

        self.runs = True

        self.time_text.text = str("Time = ")
        self.mode_text.text = ""
        self.trajectory.set_data([],[])

        x_init, y_init = self.state_log[0][0], self.state_log[1][0]
        self.init_state.set_data(x_init, y_init)

        num_eq = self.equilibria.shape[0]
        x_eq, y_eq = np.zeros(num_eq), np.zeros(num_eq)
        for k in range(num_eq):
            x_eq[k], y_eq[k] = self.equilibria[k,0], self.equilibria[k,1]
        self.equilibria_plot.set_data(x_eq, y_eq)

        for cbf in self.cbfs:
            self.cbf_contours.append( cbf.contour_plot(self.ax, levels=[0.0], colors=['green'], min=self.x_lim[0], max=self.x_lim[1], resolution=0.2) )

        graphical_elements = []
        graphical_elements.append(self.time_text)
        graphical_elements.append(self.mode_text)
        graphical_elements.append(self.trajectory)
        graphical_elements.append(self.init_state)
        graphical_elements.append(self.equilibria_plot)
        graphical_elements += self.clf_contours.collections
        for cbf_countour in self.cbf_contours:
            graphical_elements += cbf_countour.collections

        return graphical_elements

    def update(self, i):

        if i <= self.num_steps:

            xdata, ydata = self.state_log[0][0:i], self.state_log[1][0:i]
            self.trajectory.set_data(xdata, ydata)

            current_time = np.around(self.time[i], decimals = 2)
            current_state = [ self.state_log[0][i], self.state_log[1][i] ]

            if hasattr(self, 'clf_log'):
                current_piv_state = [self.clf_log[0][i], self.clf_log[1][i], self.clf_log[2][i]]
                self.clf.set_param(current_piv_state)

            self.time_text.set_text("Time = " + str(current_time) + "s")

            if self.mode_log:
                if self.mode_log[i] == 1:
                    self.mode_text.set_text("Compatibility")
                elif self.mode_log[i] == 0:
                    self.mode_text.set_text("Rate")

            if self.draw_level:
                V = self.clf.evaluate_function(*current_state)[0]
                # h = self.cbf.evaluate_function(*current_state)[0]
                for coll in self.clf_contours.collections:
                    coll.remove()
                # for coll in self.cbf_contours.collections:
                #     coll.remove()
                perimeter = 4*abs(current_state[0]) + 4*abs(current_state[1])
                self.clf_contours = self.clf.contour_plot(self.ax, levels=[V], colors=['blue'], min=self.x_lim[0], max=self.x_lim[1], resolution=0.008*perimeter+0.1)
                # self.cbf_contours = self.cbf.contour_plot(self.ax, levels=[h], colors=['green'], min=self.x_lim[0], max=self.x_lim[1], resolution=0.008*perimeter+0.1)

            # if hasattr(self, 'cbf_log'):
            #     current_pih_state = [self.cbf_log[0][i], self.cbf_log[1][i], self.cbf_log[2][i]]
            #     self.cbf.set_param(current_pih_state)

            # for coll in self.cbf_contours.collections:
                # coll.remove()
            # self.cbf_contours = self.cbf.contour_plot(self.ax, levels=[0.0], colors=['green'], min=self.x_lim[0], max=self.x_lim[1], resolution=0.2)

        else:
            self.runs = False
            self.animation.event_source.stop()

        graphical_elements = []
        graphical_elements.append(self.time_text)
        graphical_elements.append(self.mode_text)
        graphical_elements.append(self.trajectory)
        graphical_elements.append(self.init_state)
        graphical_elements.append(self.equilibria_plot)
        graphical_elements += self.clf_contours.collections
        for cbf_countour in self.cbf_contours:
            graphical_elements += cbf_countour.collections

        return graphical_elements

    def gen_function(self):
        while self.runs:
            yield self.current_step
            self.current_step += int(max(1,np.ceil(self.anim_step)))

    def animate(self):
        self.animation = anim.FuncAnimation(self.fig, func=self.update, frames=self.gen_function(), init_func=self.init, interval=1000/self.fps, repeat=False, blit=True)

    def get_frame(self, t):
        '''
        Updates frame at time time.
        '''
        self.init()
        step = np.argmin( (self.time - t)**2 )
        graphical_elements = self.update(step)

        return graphical_elements

    def plot_frame(self, t):
        '''
        Plots specific animation frame at time t.
        '''
        self.get_frame(t)