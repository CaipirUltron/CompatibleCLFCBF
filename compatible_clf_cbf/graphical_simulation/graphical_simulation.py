import math, rospy
import tf2_ros
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped, TransformStamped

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

        return self.time_text, self.trajectory, self.clf_level_set1, self.clf_level_set2, self.cbf_level_set1, self.cbf_level_set2, self.clf_crit, self.cbf_crit

    def animate(self):
        self.animation = anim.FuncAnimation(self.fig, func=self.update, init_func=self.init, frames=self.num_steps, interval=20, repeat=False, blit=True)


class SimulationRviz():
    def __init__(self, plant, clf, cbf):
        
        # Initialize important class attributes
        self._plant = plant
        self._clf = clf
        self._cbf = cbf

        # Initialize node
        rospy.init_node('graphics_broadcaster', anonymous = True)
        rospy.loginfo("Starting graphical simulation...")

        self.ref_marker_size = 0.2
        self.num_points = 100

        # ROS subscribers
        # click_subscriber = rospy.Subscriber("clicked_point", PointStamped, self.draw_reference)

        # ROS Publishers
        self.trajectory_publisher = rospy.Publisher('trajectory', Marker, queue_size=1)
        self.ref_publisher = rospy.Publisher('ref', Marker, queue_size=1)

        self.clf_publisher = rospy.Publisher('clf', MarkerArray, queue_size=1)
        self.cbf_publisher = rospy.Publisher('cbf', MarkerArray, queue_size=1)

        self.branch0_publisher = rospy.Publisher('branch0', Marker, queue_size=1)
        self.branch1_publisher = rospy.Publisher('branch1', Marker, queue_size=1)
        self.branch2_publisher = rospy.Publisher('branch2', Marker, queue_size=1)

        # Setup base transform
        self.world_transform = TransformStamped()
        self.world_transform.header.stamp = rospy.Time.now()
        self.world_transform.header.frame_id = "world"
        self.world_transform.child_frame_id = "base_frame"
        self.world_transform.transform.translation.x = 0.0
        self.world_transform.transform.translation.y = 0.0
        self.world_transform.transform.translation.z = 0.0
        self.world_transform.transform.rotation.x = 0.0
        self.world_transform.transform.rotation.y = 0.0
        self.world_transform.transform.rotation.z = 0.0
        self.world_transform.transform.rotation.w = 1.0

        # Send base transform 
        self.world_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.world_tf_broadcaster.sendTransform(self.world_transform)

        # Setup rviz graphical objects
        self.trajectory_marker = Marker()
        self.ref_pos_marker = Marker()

        self.clf_markers = MarkerArray()
        self.clf_markers.markers.append( Marker() )
        self.clf_markers.markers.append( Marker() )
        self.clf_markers.markers.append( Marker() )

        self.cbf_markers = MarkerArray()
        self.cbf_markers.markers.append( Marker() )
        self.cbf_markers.markers.append( Marker() )
        self.cbf_markers.markers.append( Marker() )

        # Two branches of the hyperbola
        self.branch0_marker = Marker()
        self.branch1_marker = Marker()
        self.branch2_marker = Marker()

        # Initialize robot position marker
        self.init_state(np.zeros(2))

        # Initialize reference marker
        self.init_reference(np.zeros(2))

        # Initialize CLF and CBF markers
        self.init_clf()
        self.init_cbf()
        self.init_invariance(self.branch0_marker)
        self.init_invariance(self.branch1_marker)
        self.init_invariance(self.branch2_marker)

    def init_state(self, state):

        self.trajectory_marker.header.frame_id = "base_frame"
        self.trajectory_marker.type = Marker.LINE_STRIP
        self.trajectory_marker.action = Marker.ADD
        self.trajectory_marker.scale.x = 0.05
        self.trajectory_marker.color.a = 1.0
        self.trajectory_marker.color.r = 0.0
        self.trajectory_marker.color.g = 0.0
        self.trajectory_marker.color.b = 0.0
        self.trajectory_marker.pose.position.x = state[0]
        self.trajectory_marker.pose.position.y = state[1]
        self.trajectory_marker.pose.position.z = 0.0
        self.trajectory_marker.pose.orientation.x = 0.0
        self.trajectory_marker.pose.orientation.y = 0.0
        self.trajectory_marker.pose.orientation.z = 0.0
        self.trajectory_marker.pose.orientation.w = 1.0

    def init_reference(self, ref):

        self.ref_pos_marker.header.frame_id = "base_frame"
        self.ref_pos_marker.type = Marker.SPHERE
        self.ref_pos_marker.action = Marker.ADD
        self.ref_pos_marker.scale.x = self.ref_marker_size
        self.ref_pos_marker.scale.y = self.ref_marker_size
        self.ref_pos_marker.scale.z = 0.1
        self.ref_pos_marker.color.a = 1.0
        self.ref_pos_marker.color.r = 1.0
        self.ref_pos_marker.color.g = 0.0
        self.ref_pos_marker.color.b = 0.0
        self.ref_pos_marker.pose.position.x = ref[0]
        self.ref_pos_marker.pose.position.y = ref[1]
        self.ref_pos_marker.pose.position.z = -0.1
        self.ref_pos_marker.pose.orientation.x = 0.0
        self.ref_pos_marker.pose.orientation.y = 0.0
        self.ref_pos_marker.pose.orientation.z = 0.0
        self.ref_pos_marker.pose.orientation.w = 1.0

    def init_clf(self):
        '''
        Initialize CLF markers.
        '''
        clf_color = [ 0.0, 0.4, 1.0 ]
        for k in range(3):
            self.clf_markers.markers[k].id = k
            self.clf_markers.markers[k].header.frame_id = "base_frame"
            self.clf_markers.markers[k].type = Marker.LINE_STRIP
            self.clf_markers.markers[k].scale.x = 0.05
            self.clf_markers.markers[k].color.a = 1.0
            self.clf_markers.markers[k].color.r = clf_color[0]
            self.clf_markers.markers[k].color.g = clf_color[1]
            self.clf_markers.markers[k].color.b = clf_color[2]
            self.clf_markers.markers[k].pose.position.z = 0.0

    def init_cbf(self):
        '''
        Initialize CBF markers.
        '''
        cbf_color = [ 0.0, 1.0, 0.0 ]
        for k in range(3):
            self.cbf_markers.markers[k].id = k
            self.cbf_markers.markers[k].header.frame_id = "base_frame"
            self.cbf_markers.markers[k].type = Marker.LINE_STRIP
            self.cbf_markers.markers[k].scale.x = 0.05
            self.cbf_markers.markers[k].color.a = 1.0
            self.cbf_markers.markers[k].color.r = cbf_color[0]
            self.cbf_markers.markers[k].color.g = cbf_color[1]
            self.cbf_markers.markers[k].color.b = cbf_color[2]
            self.clf_markers.markers[k].pose.position.z = 0.1

    def init_invariance(self, branch):

        branch.header.frame_id = "base_frame"
        branch.type = Marker.LINE_STRIP
        branch.action = Marker.ADD
        branch.scale.x = 0.02
        branch.color.a = 1.0
        branch.color.r = 1.0
        branch.color.g = 0.0
        branch.color.b = 0.0
        branch.pose.position.x = 0.0
        branch.pose.position.y = 0.0
        branch.pose.position.z = 0.5
        branch.pose.orientation.x = 0.0
        branch.pose.orientation.y = 0.0
        branch.pose.orientation.z = 0.0
        branch.pose.orientation.w = 1.0

    def draw_trajectory(self):
        '''
        Publishes state trajectory.
        '''
        state = self._plant.get_state()
        new_trajectory_point = Point()
        new_trajectory_point.x = state[0]
        new_trajectory_point.y = state[1]
        new_trajectory_point.z = 0
        self.trajectory_marker.points.append( new_trajectory_point )
        self.trajectory_publisher.publish(self.trajectory_marker)

    def draw_reference(self, ref_point):
        '''
        Publishes reference position marker.
        '''
        if isinstance(ref_point,type(PointStamped())):
            reference = np.array([ ref_point.point.x, ref_point.point.y ])
        else:
            reference = np.array([ ref_point[0], ref_point[1] ])

        self.ref_pos_marker.pose.position.x = reference[0]
        self.ref_pos_marker.pose.position.y = reference[1]
        self.ref_publisher.publish(self.ref_pos_marker)

    def draw_clf(self):
        '''
        Publishes CLF markers in Rviz.
        '''
        clf_value = self._clf.get_fvalue()
        self.draw_conic( self._clf, clf_value, self.clf_markers )
        self.clf_publisher.publish( self.clf_markers )

    def draw_cbf(self):
        '''
        Publishes CBF in Rviz.
        '''
        self.draw_conic( self._cbf, 0.0, self.cbf_markers )
        self.cbf_publisher.publish( self.cbf_markers )

    def draw_invariance(self, controller):

        pencil_eig = controller.pencil_dict["eigenvalues"]

        l1 = pencil_eig[0]
        l2 = pencil_eig[1]

        res1 = (l2-l1)/self.num_points
        res2 = (4.0*l2 - l2)/self.num_points

        self._branch1_marker.points = []
        self._branch2_marker.points = []
        for k in range(self.num_points):
            l1 += res1
            l2 += res2
            if all( np.absolute( l1 - pencil_eig ) > controller.eigen_threshold ):
                v = controller.v_values( l1 )
                state = v + controller.cbf.critical()
                self._branch1_marker.points.append( Point(x= state[0], y= state[1], z = 0.0) )
            if all( np.absolute( l2 - pencil_eig ) > controller.eigen_threshold ):
                v = controller.v_values( l2 )
                state = v + controller.cbf.critical()
                self._branch2_marker.points.append( Point(x= state[0], y= state[1], z = 0.0) )

        self.branch1_publisher.publish(self._branch1_marker)        
        self.branch2_publisher.publish(self._branch2_marker)        

    def get_reference(self):
        return self._reference

    @staticmethod
    def draw_conic(quadratic, quadratic_level, marker_array, num_points=50):
        '''
        Draws a conic section of a given quadratic function, using an Rviz Marker.
        Receives a quadratic function and a list of markers ( marker[0] - ellipse, marker[1] - hyperbola (branch 1), marker[2] (branch 2) ).
        '''
        eigs, angle, Q = quadratic.compute_eig()
        critical_pt = quadratic.get_critical()
        height = quadratic.get_height()

        ellipse_marker = marker_array.markers[0]
        hyperbola_marker1 = marker_array.markers[1]
        hyperbola_marker2 = marker_array.markers[2]

        ellipse_marker.pose.position.x = critical_pt[0]
        ellipse_marker.pose.position.y = critical_pt[1]
        ellipse_marker.pose.orientation.z = np.sin(-angle/2)
        ellipse_marker.pose.orientation.w = np.cos(-angle/2)

        hyperbola_marker1.pose.position.x = critical_pt[0]
        hyperbola_marker1.pose.position.y = critical_pt[1]
        hyperbola_marker1.pose.orientation.z = np.sin(-angle/2)
        hyperbola_marker1.pose.orientation.w = np.cos(-angle/2)

        hyperbola_marker2.pose.position.x = critical_pt[0]
        hyperbola_marker2.pose.position.y = critical_pt[1]
        hyperbola_marker2.pose.orientation.z = np.sin(-angle/2)
        hyperbola_marker2.pose.orientation.w = np.cos(-angle/2)

        scale_x = np.sqrt((2/np.abs(eigs[0])*np.abs(quadratic_level - height)))
        scale_y = np.sqrt((2/np.abs(eigs[1])*np.abs(quadratic_level - height)))
        
        res = 2*math.pi/num_points
        if eigs[0]*eigs[1] > 0:
            ellipse_marker.action = Marker.ADD
            hyperbola_marker1.action = Marker.DELETE
            hyperbola_marker2.action = Marker.DELETE

            ellipse_marker.points = []
            for k in range(num_points+1):
                ellipse_point = np.array([ scale_x*np.cos(k*res), scale_y*np.sin(k*res) ])
                ellipse_marker.points.append( Point(x=ellipse_point[0], y=ellipse_point[1], z = 0.0) )
        else:
            ellipse_marker.action = Marker.DELETE

            hyperbola_marker1.points = []
            hyperbola_marker2.points = []
            for k in range(-num_points,num_points):
                if eigs[0] < 0:
                    hyperbola_point1 = np.array( [ scale_x*np.sinh(k*res),  scale_y*np.cosh(k*res)] )
                    hyperbola_point2 = np.array( [ scale_x*np.sinh(k*res), -scale_y*np.cosh(k*res)] )
                else:
                    hyperbola_point1 = np.array( [ scale_x*np.cosh(k*res), scale_y*np.sinh(k*res)] )
                    hyperbola_point2 = np.array( [-scale_x*np.cosh(k*res), scale_y*np.sinh(k*res)] )
                hyperbola_marker1.points.append( Point(x=hyperbola_point1[0], y=hyperbola_point1[1], z = 0.0) )
                hyperbola_marker2.points.append( Point(x=hyperbola_point2[0], y=hyperbola_point2[1], z = 0.0) )