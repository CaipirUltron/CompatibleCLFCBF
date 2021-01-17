import rospy
import numpy as np

from rospy.numpy_msg import numpy_msg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped, PolygonStamped

import tf
import time
from tf.transformations import quaternion_from_euler

# Class for 2D simulations in Rviz
class GraphicalSimulation():

    def __init__(self):
        # Initialize important class attributes
        # self._clf = clf
        # self._cbf = cbf

        # ROS Publishers
        self.trajectory_publisher = rospy.Publisher('trajectory', Marker, queue_size=1)
        self.clf_publisher = rospy.Publisher('clf', Marker, queue_size=1)
        self.cbf_publisher = rospy.Publisher('cbf', Marker, queue_size=1)

        # Setup tf tree and rviz graphical objects
        self.world_tf = tf.TransformBroadcaster()
        self._trajectory_marker = Marker()
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

    def draw_trajectory(self, point):
        new_trajectory_point = Point()
        new_trajectory_point.x = point[0]
        new_trajectory_point.y = point[1]
        new_trajectory_point.z = 0
        self._trajectory_marker.points.append( new_trajectory_point )

        # Sends baseline transformation
        self.world_tf.sendTransform((0, 0, 0), quaternion_from_euler(0, 0, 0), rospy.Time.now(), "base_frame", "world")

        # Publishes Rviz graphical objects
        self.trajectory_publisher.publish(self._trajectory_marker)