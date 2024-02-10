import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as mcolors

from matplotlib import gridspec
from matplotlib.patches import Rectangle

class Plot2DSimulation():
    '''
    Class for matplotlib-based simulation of a point-like 2D dynamical system.
    '''
    def __init__(self, logs, robot, clf, cbfs, **kwargs):
        
        self.plot_config = {
            "figsize": (5,5),
            "gridspec": (1,1,1),
            "widthratios": [1],
            "heightratios": [1],
            "limits": [[-6,6], [-6,6]],
            "drawlevel": False,
            "resolution": 50,
            "fps":50,
            "pad":2.0,
            "invariants": False,
            "equilibria": False,
            "arrows": False
        }
        if "plot_config" in kwargs.keys():
            self.plot_config = kwargs["plot_config"]
        
        self.logs = logs
        self.robot = robot
        self.clf, self.cbfs = clf, cbfs

        self.fig = plt.figure(figsize = self.plot_config["figsize"], constrained_layout=True)
        self.configure()
        
        # self.fig.tight_layout(pad=plot_config["pad"])
        self.animation = None

        print("Equilibrium pts: ")
        for eq in self.equilibria:
            print(eq)

    def configure(self):
        '''
        Configure axes.
        '''
        self.ax_struct = self.plot_config["gridspec"][0:2]
        width_ratios = self.plot_config["widthratios"]
        height_ratios = self.plot_config["heightratios"]
        gs = gridspec.GridSpec(self.ax_struct[0], self.ax_struct[1], width_ratios = width_ratios, height_ratios = height_ratios)

        # Specify main ax
        main_ax = self.plot_config["gridspec"][-1]

        '''
        I honestly don't know wtf is this
        '''
        def order2indexes(k, m):
            i = int((k-1)/m)
            j = int(np.mod(k-1, m))
            return i,j

        if isinstance(main_ax, list):
            i_list, j_list = [], []
            for axes in main_ax:
                i,j = order2indexes(axes, self.ax_struct[1])
                i_list.append(i)
                j_list.append(j)
            i = np.sort(i_list).tolist()
            j = np.sort(j_list).tolist()
            if i[0] == i[-1]: i = i[0]
            if j[0] == j[-1]: j = j[0]
        
        if isinstance(main_ax, int):
            i,j = order2indexes(main_ax, self.ax_struct[1])

        if isinstance(i, int):
            if isinstance(j, int):
                self.main_ax = self.fig.add_subplot(gs[i,j])
            else:
                self.main_ax = self.fig.add_subplot(gs[i,j[0]:(j[-1]+1)])
        else:
            if isinstance(j, int):
                self.main_ax = self.fig.add_subplot(gs[i[0]:(i[-1]+1),j])
            else:
                self.main_ax = self.fig.add_subplot(gs[i[0]:(i[-1]+1),j[0]:(j[-1]+1)])

        '''
        Configures plot
        '''
        self.x_lim = self.plot_config["limits"][0]
        self.y_lim = self.plot_config["limits"][1]

        self.main_ax.set_xlim(*self.x_lim)
        self.main_ax.set_ylim(*self.y_lim)
        self.main_ax.set_title("", fontsize=18)
        self.main_ax.set_aspect('equal', adjustable='box')

        self.draw_level = self.plot_config["drawlevel"]
        self.fps = self.plot_config["fps"]
        self.numpoints = self.plot_config["resolution"]

        self.create_graphical_objs()

    def create_graphical_objs(self):
        '''
        Create graphical objects into the correct axes.
        '''
        # Get logs
        self.sample_time = self.logs["sample_time"]
        self.time = np.array(self.logs["time"])
        self.state_log = self.logs["state"]
        if "clf_log" in self.logs.keys():
            self.clf_log = self.logs["clf_log"]
            self.clf_param_dim = np.shape(np.array(self.clf_log))[0]

        # self.mode_log = self.logs["mode"]
        self.equilibria = self.logs["equilibria"]
        self.num_steps = len(self.state_log[0])
        self.anim_step = (self.num_steps/self.time[-1])/self.fps
        self.current_step = 0
        self.runs = False

        self.num_cbfs = len(self.cbfs)

        # Initalize some graphical objects
        self.time_text = self.main_ax.text(0.5, self.y_lim[0]+0.5, str("Time = "), fontsize=14)
        self.trajectory, = self.main_ax.plot([],[],lw=2)
        self.init_state, = self.main_ax.plot([],[],'bo',lw=2)
        self.actual_state, = self.main_ax.plot([],[],'bo',lw=1,alpha=0.5)
        self.equilibria_plot, = self.main_ax.plot([],[], marker='*', mfc='none', lw=2, color='k', linestyle="None")
        # self.equilibria_plot, = self.main_ax.plot([],[], marker='o', color='r',lw=2)
        self.clf_grad_arrow, = self.main_ax.plot([],[],'b',lw=0.8)
        self.cbf_grad_arrow, = self.main_ax.plot([],[],'r',lw=0.8)
        self.fc_arrow, = self.main_ax.plot([],[],'r',lw=0.8)

        self.clf_contour_color = mcolors.TABLEAU_COLORS['tab:blue']
        self.cbf_contour_color = mcolors.TABLEAU_COLORS['tab:red']

        self.clf_contours = self.clf.plot_levels(levels=[0.0], colors=self.clf_contour_color, ax=self.main_ax, limits=self.plot_config["limits"], spacing=0.5)
        self.cbf_contours = []

    def init(self):

        self.runs = True

        self.time_text.text = str("Time = ")
        # self.mode_text.text = ""
        self.trajectory.set_data([],[])

        x_init, y_init = self.state_log[0][0], self.state_log[1][0]
        self.init_state.set_data(x_init, y_init)

        if self.plot_config["equilibria"]:
            num_eq = len(self.equilibria)
            x_eq, y_eq = np.zeros(num_eq), np.zeros(num_eq)
            for k in range(num_eq):
                x_eq[k], y_eq[k] = self.equilibria[k]["x"][0], self.equilibria[k]["x"][1]
            self.equilibria_plot.set_data(x_eq, y_eq)

        for cbf in self.cbfs:
            self.cbf_contours.append( cbf.plot_levels(levels=[-0.1*k for k in range(4,-1,-1)], colors=self.cbf_contour_color, ax=self.main_ax, limits=self.plot_config["limits"], spacing=0.1) )

        graphical_elements = []
        graphical_elements.append(self.time_text)
        graphical_elements.append(self.trajectory)
        graphical_elements.append(self.init_state)
        graphical_elements.append(self.actual_state)
        graphical_elements.append(self.equilibria_plot)
        graphical_elements.append(self.clf_grad_arrow)
        graphical_elements.append(self.cbf_grad_arrow)
        graphical_elements.append(self.fc_arrow)
        graphical_elements += self.clf_contours
        for cbf_countour in self.cbf_contours:
            graphical_elements += cbf_countour

        return graphical_elements

    def update(self, i):

        if i <= self.num_steps:

            xdata, ydata = self.state_log[0][0:i], self.state_log[1][0:i]
            self.trajectory.set_data(xdata, ydata)
            if len(xdata) > 0:
                self.actual_state.set_data(xdata[-1], ydata[-1])

            current_time = np.around(self.time[i], decimals = 2)
            current_state = [ self.state_log[0][i], self.state_log[1][i] ]

            if self.plot_config["arrows"]:
                nablaV = self.clf.gradient(current_state)
                nablaV_norm = nablaV/np.linalg.norm(nablaV)
                self.clf_grad_arrow.set_data( 
                    [ current_state[0], current_state[0] + nablaV_norm[0] ], 
                    [ current_state[1], current_state[1] + nablaV_norm[1] ] )

                nablah = self.cbfs[0].gradient(current_state)
                nablah_norm = nablah/np.linalg.norm(nablah)
                self.cbf_grad_arrow.set_data( 
                    [ current_state[0], current_state[0] + nablah_norm[0] ], 
                    [ current_state[1], current_state[1] + nablah_norm[1] ] )

                F = self.robot.get_F()
                fc = np.array([F[1,0], F[2,0]])
                fc_norm = fc/np.linalg.norm(fc)
                self.fc_arrow.set_data( 
                    [ current_state[0], current_state[0] + fc_norm[0] ], 
                    [ current_state[1], current_state[1] + fc_norm[1] ] )
            
            if hasattr(self, 'clf_log'):
                current_piv_state = [ self.clf_log[k][i] for k in range(self.clf_param_dim) ]
                self.clf.set_param(current_piv_state)

            self.time_text.set_text("Time = " + str(current_time) + "s")

            # if len(self.equilibria) != 0 and self.equilibria[i] != None:
            #     x_eq, y_eq = self.equilibria[i][0], self.equilibria[i][1]
            #     self.equilibria_plot.set_data(x_eq, y_eq)

            if self.draw_level:
                V = self.clf.function(current_state)
                for coll in self.clf_contours:
                    coll.remove()

                # perimeter = 4*abs(current_state[0]) + 4*abs(current_state[1])
                # self.clf_contours = self.clf.plot_levels(levels=[V], colors=self.clf_contour_color, ax=self.main_ax, limits=self.plot_config["limits"], spacing=0.005*perimeter+0.1)
                self.clf_contours = self.clf.plot_levels(levels=[V], colors=self.clf_contour_color, ax=self.main_ax, limits=self.plot_config["limits"], spacing=0.5)

        else:
            self.runs = False
            # self.animation.event_source.stop()
            
        graphical_elements = []
        graphical_elements.append(self.time_text)
        graphical_elements.append(self.trajectory)
        graphical_elements.append(self.init_state)
        graphical_elements.append(self.actual_state)
        graphical_elements.append(self.equilibria_plot)
        graphical_elements.append(self.clf_grad_arrow)
        graphical_elements.append(self.cbf_grad_arrow)
        graphical_elements.append(self.fc_arrow)
        graphical_elements += self.clf_contours
        for cbf_countour in self.cbf_contours:
            graphical_elements += cbf_countour

        return graphical_elements

    def gen_function(self, initial_step):
        self.current_step = initial_step
        while self.runs:
            yield self.current_step
            self.current_step += int(max(1,np.ceil(self.anim_step)))
        final_step = int(np.floor(self.time[-1]/self.sample_time))
        yield final_step

    def animate(self, *args):
        '''
        Show animation starting from specific time (passed as optional argument)
        '''
        initial_time = 0
        if len(args) > 0:
            initial_time = args[0]
        initial_step = int(np.floor(initial_time/self.sample_time))
        self.animation = anim.FuncAnimation(self.fig, func=self.update, frames=self.gen_function(initial_step), init_func=self.init, interval=1000/self.fps, repeat=False, blit=True, cache_frame_data=False)

    def get_frame(self, t):
        '''
        Returns graphical elements at time t.
        '''
        self.init()
        step = int(np.floor(t/self.sample_time))
        graphical_elements = self.update(step)

        return graphical_elements

    def plot_frame(self, t):
        '''
        Updates frame at time t.
        '''
        self.get_frame(t)

class PlotPFSimulation(Plot2DSimulation):
    '''
    Class for matplotlib-based simulation of the vehicle performing path following.
    '''
    def __init__(self, path, logs, robot, clf, cbfs, **kwargs):
        super().__init__(logs, robot, clf, cbfs, **kwargs)
        self.path = path

    def create_graphical_objs(self):
        super().create_graphical_objs()

        self.gamma_log = self.logs["gamma_log"]
        self.path_length = self.plot_config["path_length"]
        self.path_color = mcolors.TABLEAU_COLORS['tab:red']
        self.path_graph, self.virtual_pts = [], []
        self.path_graph, = self.main_ax.plot([],[], linestyle='dashed', lw=1.2, alpha=0.8, color=self.path_color)
        self.virtual_pt,  = self.main_ax.plot([],[],lw=1,color='red',marker='o',markersize=4.0)

    def init(self):
        graphical_elements = super().init()

        self.robot_pos, = self.main_ax.plot([],[],lw=1,color='b',marker='o',markersize=4.0)

        xpath, ypath = [], []
        for k in range(self.numpoints):
            gamma = k*self.path_length/self.numpoints
            pos = self.path.get_path_point(gamma)
            xpath.append(pos[0])
            ypath.append(pos[1])
        self.path_graph.set_data(xpath, ypath)
        self.virtual_pt.set_data([],[])

        graphical_elements.append(self.path_graph)
        graphical_elements.append(self.virtual_pt)
        graphical_elements.append(self.robot_pos)

        return graphical_elements
    
    def update(self, i):

        if len(self.gamma_log) > 0:
            gamma = self.gamma_log[i]
            pos = self.path.get_path_point(gamma)
            self.virtual_pt.set_data(pos[0], pos[1])
            self.clf.set_critical( pos )

        self.robot_pos.set_data(self.state_log[0][i], self.state_log[1][i])

        graphical_elements = super().update(i)

        graphical_elements.append(self.path_graph)
        graphical_elements.append( self.virtual_pt )
        graphical_elements.append(self.robot_pos)

        return graphical_elements
    
class PlotUnicycleSimulation(Plot2DSimulation):
    '''
    Class for matplotlib-based simulation of the unicycle robot.
    '''
    def __init__(self, logs, robot, clf, cbfs, **kwargs):
        super().__init__(logs, robot, clf, cbfs, **kwargs)

    def create_graphical_objs(self):
        '''
        Create graphical objects into the correct axes.
        '''
        # Get logs
        self.time = np.array(self.logs["time"])
        self.state_log = self.logs["state"]
        if "clf_log" in self.logs.keys():
            self.clf_log = self.logs["clf_log"]
            self.clf_param_dim = np.shape(np.array(self.clf_log))[0]
        # self.mode_log = self.logs["mode"]
        self.equilibria = np.array(self.logs["equilibria"])
        self.num_steps = len(self.state_log[0])
        self.anim_step = (self.num_steps/self.time[-1])/self.fps
        self.current_step = 0
        self.runs = False

        self.num_cbfs = len(self.cbfs)

        # Initalize some graphical objects
        self.time_text = self.main_ax.text(0.5, self.y_lim[0]+0.5, str("Time = "), fontsize=14)
        self.cbf_text_numbers = [ self.main_ax.text(0, 0, "", fontsize=10) for k in range(self.num_cbfs) ]
        self.cbf_text_numbers_closests = [ self.main_ax.text(0, 0, "", fontsize=10) for k in range(self.num_cbfs) ]

        # self.origin, = self.main_ax.plot([],[],lw=4, marker='*', color=[0.,0.,0.])
        self.trajectory, = self.main_ax.plot([],[],lw=2)
        self.init_state, = self.main_ax.plot([],[],'bo',lw=2)
        self.equilibria_plot, = self.main_ax.plot([],[], marker='o', mfc='none', lw=2, color=[1.,0.,0.], linestyle="None")
        self.closests = self.main_ax.scatter([],[], marker='o', color=mcolors.TABLEAU_COLORS['tab:green'])

        self.robot_color = mcolors.TABLEAU_COLORS['tab:blue']
        self.clf_contour_color = mcolors.TABLEAU_COLORS['tab:blue']
        self.cbf_contour_color = mcolors.TABLEAU_COLORS['tab:green']

        self.clf_contours = self.clf.contour_plot(self.main_ax, levels=[0.0], colors=self.clf_contour_color, min=self.x_lim[0], max=self.x_lim[1], resolution=0.5)
        self.cbf_contours = []

        robot_x = self.state_log[0][0]
        robot_y = self.state_log[1][0]
        robot_angle = self.state_log[2][0]
        center = self.robot.geometry.get_center( (robot_x, robot_y, robot_angle) )
        self.robot_geometry = Rectangle( center,
                                width = self.robot.geometry.length,
                                height = self.robot.geometry.width, 
                                angle=np.rad2deg(robot_angle), rotation_point="xy", color = self.robot_color )

        self.robot_pos, = self.main_ax.plot([],[],lw=1,color='black',marker='o',markersize=4.0)
        self.angle_line, = self.main_ax.plot([],[],lw=2, color = mcolors.TABLEAU_COLORS['tab:green'])

        self.circle = plt.Circle((robot_x, robot_y), self.plot_config["radius"], color=mcolors.TABLEAU_COLORS['tab:green'], linestyle = '--', fill=False)

    def update(self, i):

        if i <= self.num_steps:

            xdata, ydata = self.state_log[0][0:i], self.state_log[1][0:i]
            self.trajectory.set_data(xdata, ydata)
            current_time = np.around(self.time[i], decimals = 2)
            current_position = [ self.state_log[0][i], self.state_log[1][i] ]

            robot_x = self.state_log[0][i]
            robot_y = self.state_log[1][i]
            robot_angle = self.state_log[2][i]

            self.robot_pos.set_data(robot_x, robot_y)

            pose = (robot_x, robot_y, robot_angle)
            current_center = self.robot.geometry.get_center( pose )

            self.robot_geometry.xy = self.robot.geometry.get_corners(pose, "bottomleft")
            self.robot_geometry.angle = np.rad2deg(robot_angle)
            self.circle.center = current_center

            self.main_ax.add_patch(self.robot_geometry)
            self.main_ax.add_patch(self.circle)

            if hasattr(self, 'clf_log'):
                current_piv_state = [ self.clf_log[k][i] for k in range(self.clf_param_dim) ]
                self.clf.set_param(current_piv_state)

            self.time_text.set_text("Time = " + str(current_time) + "s")

            if self.draw_level:
                V = self.clf.evaluate_function(*current_position)[0]
                for coll in self.clf_contours.collections:
                    coll.remove()
                perimeter = 4*abs(current_position[0]) + 4*abs(current_position[1])
                self.clf_contours = self.clf.contour_plot(self.main_ax, levels=[V], colors=self.clf_contour_color, min=self.x_lim[0], max=self.x_lim[1], resolution=0.008*perimeter+0.1)

            data = np.zeros([self.num_cbfs, 2])
            for k in range(self.num_cbfs):
                self.cbf_text_numbers[k].set_text( str(k) )
                self.cbf_text_numbers[k].set_position( self.cbfs[k].get_critical().tolist() )

                h, nablah, closest_pt, gamma_opt = self.cbfs[k].barrier_set({"radius": self.plot_config["radius"], "center": current_center, "orientation": robot_angle})

                self.cbf_text_numbers_closests[k].set_text( str(k) )
                self.cbf_text_numbers_closests[k].set_position( closest_pt )

                data[k,0], data[k,1] = closest_pt[0], closest_pt[1]
            self.closests.set_offsets(data)
        else:
            self.runs = False
            self.animation.event_source.stop()

        graphical_elements = []
        graphical_elements.append(self.time_text)
        # graphical_elements.append(self.mode_text)
        graphical_elements.append(self.trajectory)
        graphical_elements.append(self.init_state)
        graphical_elements.append(self.equilibria_plot)
        graphical_elements.append(self.robot_geometry)
        graphical_elements.append(self.robot_pos)
        graphical_elements.append(self.circle)
        graphical_elements.append(self.closests)

        graphical_elements += self.cbf_text_numbers
        graphical_elements += self.cbf_text_numbers_closests

        graphical_elements += self.clf_contours.collections
        for cbf_countour in self.cbf_contours:
            graphical_elements += cbf_countour.collections

        return graphical_elements