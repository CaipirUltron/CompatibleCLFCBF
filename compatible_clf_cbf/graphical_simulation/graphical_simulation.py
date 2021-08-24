import rospy
import tf2_ros
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from visualization_msgs.msg import Marker
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
        self.state_log = logs["stateLog"]
        self.clf_log = logs["clfLog"]
        self.num_steps = len(self.state_log[0])

        # Get point resolution for graphical objects
        self.numpoints = numpoints

        self.clf, self.cbf = clf, cbf

        # Initalize graphical objects
        self.trajectory, = self.ax.plot([],[],lw=2)
        self.clf_level_set, = self.ax.plot([],[],'b',lw=1)
        self.cbf_level_set, = self.ax.plot([],[],'g',lw=1)
        # self.clf_arrows = self.ax.quiver([0.0],[0.0],[0.1],[0.1],pivot='mid',color='b')
        # self.cbf_arrows = self.ax.quiver([0.0],[0.0],[0.1],[0.1],pivot='mid',color='g')

        self.animation = None

    def init(self):

        self.trajectory.set_data([],[])
        self.clf_level_set.set_data([],[])
        self.cbf_level_set.set_data([],[])

        return self.trajectory, self.clf_level_set, self.cbf_level_set

    def update(self, i):

        xdata, ydata = self.state_log[0][0:i], self.state_log[1][0:i]
        self.trajectory.set_data(xdata, ydata)

        current_state = np.array([self.state_log[0][i], self.state_log[1][i]])
        current_pi_state = np.array([self.clf_log[0][i], self.clf_log[1][i], self.clf_log[2][i]])

        Hv = Quadratic.vector2sym(current_pi_state)
        self.clf.set_param(hessian=Hv)
        
        if self.draw_level:
            V = self.clf(current_state)
            xclf, yclf, uclf, vclf = self.clf.superlevel(V, self.numpoints)
            self.clf_level_set.set_data(xclf, yclf)

        h = self.cbf(current_state)
        xcbf, ycbf, ucbf, vcbf = self.cbf.superlevel(0.5, self.numpoints)
        self.cbf_level_set.set_data(xcbf, ycbf)

        if self.draw_gradient:
        # self.clf_arrows = self.ax.quiver(xclf, yclf, uclf, vclf, pivot='tail', color='b', scale=50.0, headlength=0.5, headwidth=1.0)
            self.cbf_arrows = self.ax.quiver(xcbf, ycbf, ucbf, vcbf, pivot='tail', color='g', scale=50.0, headlength=0.5, headwidth=1.0)

        return self.trajectory, self.clf_level_set, self.cbf_level_set,

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
        self.num_invariance_points = 50

        # ROS subscribers
        # click_subscriber = rospy.Subscriber("clicked_point", PointStamped, self.draw_reference)

        # ROS Publishers
        self.trajectory_publisher = rospy.Publisher('trajectory', Marker, queue_size=1)
        self.ref_publisher = rospy.Publisher('ref', Marker, queue_size=1)
        self.clf_publisher = rospy.Publisher('clf', Marker, queue_size=1)
        self.cbf_publisher = rospy.Publisher('cbf', Marker, queue_size=1)
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
        self._trajectory_marker = Marker()
        self._ref_pos_marker = Marker()
        self._clf_marker = Marker()
        self._cbf_marker = Marker()

        # Two branches of the hyperbola
        self._branch0_marker = Marker()
        self._branch1_marker = Marker()
        self._branch2_marker = Marker()

        # Initialize robot position marker
        self.init_state(np.zeros(2))

        # Initialize reference marker
        self.init_reference(np.zeros(2))

        # Initialize CLF and CBF markers
        self.init_clf(clf)
        self.init_cbf(cbf)
        self.init_invariance(self._branch0_marker)
        self.init_invariance(self._branch1_marker)
        self.init_invariance(self._branch2_marker)

    def init_state(self, state):

        self._trajectory_marker.header.frame_id = "base_frame"
        self._trajectory_marker.type = self._trajectory_marker.LINE_STRIP
        self._trajectory_marker.action = self._trajectory_marker.ADD
        self._trajectory_marker.scale.x = 0.05
        self._trajectory_marker.color.a = 1.0
        self._trajectory_marker.color.r = 0.0
        self._trajectory_marker.color.g = 0.5
        self._trajectory_marker.color.b = 1.0
        self._trajectory_marker.pose.position.x = state[0]
        self._trajectory_marker.pose.position.y = state[1]
        self._trajectory_marker.pose.position.z = 0.0
        self._trajectory_marker.pose.orientation.x = 0.0
        self._trajectory_marker.pose.orientation.y = 0.0
        self._trajectory_marker.pose.orientation.z = 0.0
        self._trajectory_marker.pose.orientation.w = 1.0

    def init_reference(self, ref):

        self._ref_pos_marker.header.frame_id = "base_frame"
        self._ref_pos_marker.type = self._ref_pos_marker.SPHERE
        self._ref_pos_marker.action = self._ref_pos_marker.ADD
        self._ref_pos_marker.scale.x = self.ref_marker_size
        self._ref_pos_marker.scale.y = self.ref_marker_size
        self._ref_pos_marker.scale.z = 0.1
        self._ref_pos_marker.color.a = 1.0
        self._ref_pos_marker.color.r = 1.0
        self._ref_pos_marker.color.g = 0.0
        self._ref_pos_marker.color.b = 0.0
        self._ref_pos_marker.pose.position.x = ref[0]
        self._ref_pos_marker.pose.position.y = ref[1]
        self._ref_pos_marker.pose.position.z = -0.1
        self._ref_pos_marker.pose.orientation.x = 0.0
        self._ref_pos_marker.pose.orientation.y = 0.0
        self._ref_pos_marker.pose.orientation.z = 0.0
        self._ref_pos_marker.pose.orientation.w = 1.0

    def init_clf(self, clf):
        
        self.clf_lambda, self.clf_angle, _ = clf.compute_eig()
        self._clf_marker.header.frame_id = "base_frame"
        self._clf_marker.type = self._clf_marker.SPHERE
        self._clf_marker.action = self._clf_marker.ADD
        self._clf_marker.scale.x = 2*np.sqrt(1/self.clf_lambda[0])
        self._clf_marker.scale.y = 2*np.sqrt(1/self.clf_lambda[1])
        self._clf_marker.scale.z = -0.5
        self._clf_marker.color.a = 0.3
        self._clf_marker.color.r = 0.0
        self._clf_marker.color.g = 0.4
        self._clf_marker.color.b = 1.0
        self._clf_marker.pose.position.x = clf.critical_point[0]
        self._clf_marker.pose.position.y = clf.critical_point[1]
        self._clf_marker.pose.position.z = 0
        self._clf_marker.pose.orientation.x = 0.0
        self._clf_marker.pose.orientation.y = 0.0
        self._clf_marker.pose.orientation.z = np.sin(-self.clf_angle/2)
        self._clf_marker.pose.orientation.w = np.cos(-self.clf_angle/2)

    def init_cbf(self, cbf):

        self.cbf_lambda, self.cbf_angle, _ = cbf.compute_eig()
        self._cbf_marker.header.frame_id = "base_frame"
        self._cbf_marker.type = self._cbf_marker.SPHERE
        self._cbf_marker.action = self._clf_marker.ADD
        self._cbf_marker.scale.x = 2*np.sqrt(1/self.cbf_lambda[0])
        self._cbf_marker.scale.y = 2*np.sqrt(1/self.cbf_lambda[1])
        self._cbf_marker.scale.z = 0.1
        self._cbf_marker.color.a = 0.3
        self._cbf_marker.color.r = 0.0
        self._cbf_marker.color.g = 1.0
        self._cbf_marker.color.b = 0.0
        self._cbf_marker.pose.position.x = self._cbf.critical_point[0]
        self._cbf_marker.pose.position.y = self._cbf.critical_point[1]
        self._cbf_marker.pose.position.z = 0
        self._cbf_marker.pose.orientation.x = 0.0
        self._cbf_marker.pose.orientation.y = 0.0
        self._cbf_marker.pose.orientation.z = np.sin(-self.cbf_angle/2)
        self._cbf_marker.pose.orientation.w = np.cos(-self.cbf_angle/2)

    def init_invariance(self, branch):

        branch.header.frame_id = "base_frame"
        branch.type = branch.LINE_STRIP
        branch.action = branch.ADD
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
        self._trajectory_marker.points.append( new_trajectory_point )
        self.trajectory_publisher.publish(self._trajectory_marker)

    def draw_reference(self, ref_point):
        '''
        Publishes reference position marker.
        '''
        if isinstance(ref_point,type(PointStamped())):
            reference = np.array([ ref_point.point.x, ref_point.point.y ])
        else:
            reference = np.array([ ref_point[0], ref_point[1] ])

        self._ref_pos_marker.pose.position.x = reference[0]
        self._ref_pos_marker.pose.position.y = reference[1]
        self.ref_publisher.publish(self._ref_pos_marker)

    def draw_clf(self):
        '''
        Publishes CLF in Rviz.
        '''
        clf_lambda, clf_angle, _ = self._clf.compute_eig()
        state = self._plant.get_state()

        V_threshold = 0.01
        V_point = self._clf.evaluate(state)
        if V_point > V_threshold:
            bar_clf_lambda = clf_lambda/V_point
        else:
            bar_clf_lambda = clf_lambda/V_threshold

        self._clf_marker.scale.x = 2*np.sqrt(2/bar_clf_lambda[0])
        self._clf_marker.scale.y = 2*np.sqrt(2/bar_clf_lambda[1])
        self._clf_marker.pose.position.x = self._clf.critical_point[0]
        self._clf_marker.pose.position.y = self._clf.critical_point[1]
        self._clf_marker.pose.orientation.z = np.sin(-clf_angle/2)
        self._clf_marker.pose.orientation.w = np.cos(-clf_angle/2)

        self.clf_publisher.publish(self._clf_marker)

    def draw_cbf(self):
        '''
        Publishes CBF in Rviz.
        '''
        self.cbf_publisher.publish(self._cbf_marker)

    def draw_invariance(self, controller):

        pencil_eig = controller.pencil_dict["eigenvalues"]

        l1 = pencil_eig[0]
        l2 = pencil_eig[1]

        res1 = (l2-l1)/self.num_invariance_points
        res2 = (4.0*l2 - l2)/self.num_invariance_points

        self._branch1_marker.points = []
        self._branch2_marker.points = []
        for k in range(self.num_invariance_points):
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