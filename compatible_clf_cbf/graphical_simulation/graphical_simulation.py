import rospy
import numpy as np

from rospy.numpy_msg import numpy_msg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped, PolygonStamped, PointStamped, TransformStamped

import tf, tf2_ros
import time
from tf.transformations import quaternion_from_euler

# Class for 2D simulations in Rviz
class GraphicalSimulation():

    def __init__(self, ref, *args):
        # Initialize important class attributes
        # self._clf = clf
        # self._cbf = cbf

        # Initialize node
        rospy.init_node('graphics_broadcaster', anonymous = True)
        rospy.loginfo("Starting graphical simulation...")

        self.ref_marker_size = 0.2
        for arg_index in range(len(args)):
            self.ref_marker_size = args[arg_index]

        # ROS subscribers
        click_subscriber = rospy.Subscriber("clicked_point", PointStamped, self.set_reference)

        # ROS Publishers
        self.trajectory_publisher = rospy.Publisher('trajectory', Marker, queue_size=1)
        self.ref_publisher = rospy.Publisher('ref', Marker, queue_size=1)
        self.clf_publisher = rospy.Publisher('clf', Marker, queue_size=1)
        self.cbf_publisher = rospy.Publisher('cbf', Marker, queue_size=1)

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

        # Initialize robot position marker
        self._trajectory_marker.header.frame_id = "base_frame"
        self._trajectory_marker.type = self._trajectory_marker.LINE_STRIP
        self._trajectory_marker.action = self._trajectory_marker.ADD
        self._trajectory_marker.scale.x = 0.05
        self._trajectory_marker.color.a = 1.0
        self._trajectory_marker.color.r = 0.0
        self._trajectory_marker.color.g = 0.5
        self._trajectory_marker.color.b = 1.0
        self._trajectory_marker.pose.position.x = 0.0
        self._trajectory_marker.pose.position.y = 0.0
        self._trajectory_marker.pose.position.z = 0.0
        self._trajectory_marker.pose.orientation.x = 0.0
        self._trajectory_marker.pose.orientation.y = 0.0
        self._trajectory_marker.pose.orientation.z = 0.0
        self._trajectory_marker.pose.orientation.w = 1.0

        # Initialize reference marker
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
        self._ref_pos_marker.pose.position.x = 0.0
        self._ref_pos_marker.pose.position.y = 0.0
        self._ref_pos_marker.pose.position.z = -0.1
        self._ref_pos_marker.pose.orientation.x = 0.0
        self._ref_pos_marker.pose.orientation.y = 0.0
        self._ref_pos_marker.pose.orientation.z = 0.0
        self._ref_pos_marker.pose.orientation.w = 1.0

        # Load reference
        self._reference = ref

    def draw_trajectory(self, point):
        new_trajectory_point = Point()
        new_trajectory_point.x = point[0]
        new_trajectory_point.y = point[1]
        new_trajectory_point.z = 0
        self._trajectory_marker.points.append( new_trajectory_point )

        # Sends baseline transformation
        # self.world_tf.sendTransform((point[0], point[1], 0), quaternion_from_euler(0, 0, 0), rospy.Time.now(), "robot_frame", "world")

        # Publishes Rviz graphical objects
        self.trajectory_publisher.publish(self._trajectory_marker)

    def set_reference(self, ref_point):        
        if isinstance(ref_point,type(PointStamped())):
            self._reference[0] = ref_point.point.x
            self._reference[1] = ref_point.point.y
        else:
            self._reference[0] = ref_point[0]
            self._reference[1] = ref_point[1]

        self._ref_pos_marker.pose.position.x = self._reference[0]
        self._ref_pos_marker.pose.position.y = self._reference[1]

        # Sends baseline transformation
        # self.world_tf.sendTransform((self._reference[0], self._reference[1], 0), quaternion_from_euler(0, 0, 0), rospy.Time.now(), "ref_frame", "world")

        # Publishes Rviz graphical objects
        self.ref_publisher.publish(self._ref_pos_marker)

    def get_reference(self):
        return self._reference