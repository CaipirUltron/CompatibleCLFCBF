#!/usr/bin/env python3
import rospy
import numpy as np
import tf

from rospy.numpy_msg import numpy_msg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseStamped, PolygonStamped
from tf.transformations import quaternion_from_euler

from world import World
import time

class RvizSimulation():

    def __init__(self, world):

        # Get map size
        self.world = world

        # ROS Subscribers
        # click_sub = rospy.Subscriber("clicked_point", PointStamped, update_reference)
        self.robot_sub = rospy.Subscriber("robot_state", Pose, self.draw_robot)

        # ROS Publishers
        self.world_walls_publisher = rospy.Publisher('world_walls', PolygonStamped, queue_size=1)
        self.round_obstacles_publisher = rospy.Publisher('round_obstacles', Marker, queue_size=1)
        self.robot_marker_publisher = rospy.Publisher('robot_marker', Marker, queue_size=1)
        self.angle_marker_publisher = rospy.Publisher('angle_marker', PoseStamped, queue_size=1)
        self.goal_marker_publisher = rospy.Publisher('goal_marker', Marker, queue_size=1)

        # Setup tf tree and rviz objects
        self.world_tf = tf.TransformBroadcaster()
        self.world_walls = PolygonStamped()
        self.round_obstacles = Marker()
        self.robot_position = Marker()
        self.goal_position = Marker()
        self.robot_angle = PoseStamped()
    
        # Initialize world polygon
        self.world_walls.header.frame_id = "map_frame"
        self.world_walls.polygon.points = world._corners
        for i in range(len(self.world_walls.polygon.points)):
            self.world_walls.polygon.points[i].z = 0

        # Initialize world obstacles
        self.round_obstacles.header.frame_id = "map_frame"
        self.round_obstacles.type = self.round_obstacles.SPHERE_LIST
        self.round_obstacles.action = self.round_obstacles.ADD
        self.round_obstacles.scale.x = 2*self.world.obstacles[0].radius
        self.round_obstacles.scale.y = 2*self.world.obstacles[0].radius
        self.round_obstacles.scale.z = 0.1
        self.round_obstacles.color.a = 1.0
        self.round_obstacles.color.r = 0.0
        self.round_obstacles.color.g = 1.0
        self.round_obstacles.color.b = 0.0
        self.round_obstacles.pose.position.x = 0.0
        self.round_obstacles.pose.position.y = 0.0
        self.round_obstacles.pose.position.z = -0.1
        self.round_obstacles.pose.orientation.x = 0.0
        self.round_obstacles.pose.orientation.y = 0.0
        self.round_obstacles.pose.orientation.z = 0.0
        self.round_obstacles.pose.orientation.w = 1.0
        for obstacle in self.world.obstacles:
            round_obstacle = Point()
            round_obstacle.x = obstacle.x
            round_obstacle.y = obstacle.y
            round_obstacle.z = 0
            self.round_obstacles.points.append( round_obstacle )

        # Initialize robot position marker
        self.robot_position.header.frame_id = "robot_frame"
        self.robot_position.type = self.robot_position.CYLINDER
        self.robot_position.action = self.robot_position.ADD
        self.robot_position.scale.x = 2*self.world.robot.radius
        self.robot_position.scale.y = 2*self.world.robot.radius
        self.robot_position.scale.z = 0.1
        self.robot_position.color.a = 1.0
        self.robot_position.color.r = 0.0
        self.robot_position.color.g = 0.0
        self.robot_position.color.b = 1.0
        self.robot_position.pose.position.x = 0.0
        self.robot_position.pose.position.y = 0.0
        self.robot_position.pose.position.z = -0.1
        self.robot_position.pose.orientation.x = 0.0
        self.robot_position.pose.orientation.y = 0.0
        self.robot_position.pose.orientation.z = 0.0
        self.robot_position.pose.orientation.w = 1.0

        # Initialize robot orientation marker
        self.robot_angle.header.frame_id = "robot_frame"
        self.robot_angle.pose.position.x = 0.0
        self.robot_angle.pose.position.y = 0.0
        self.robot_angle.pose.position.z = 0.0
        self.robot_angle.pose.orientation.x = 0.0
        self.robot_angle.pose.orientation.y = 0.0
        self.robot_angle.pose.orientation.z = 0.0
        self.robot_angle.pose.orientation.w = 1.0

        # Initialize goal marker
        self.goal_position.header.frame_id = "map_frame"
        self.goal_position.type = self.goal_position.SPHERE
        self.goal_position.action = self.goal_position.ADD
        self.goal_position.scale.x = 2*self.world.goal.radius
        self.goal_position.scale.y = 2*self.world.goal.radius
        self.goal_position.scale.z = 0.1
        self.goal_position.color.a = 1.0
        self.goal_position.color.r = 1.0
        self.goal_position.color.g = 0.0
        self.goal_position.color.b = 0.0
        self.goal_position.pose.position.x = 0.0
        self.goal_position.pose.position.y = 0.0
        self.goal_position.pose.position.z = -0.1
        self.goal_position.pose.orientation.x = 0.0
        self.goal_position.pose.orientation.y = 0.0
        self.goal_position.pose.orientation.z = 0.0
        self.goal_position.pose.orientation.w = 1.0

        # Initial map and robot update
        self.draw_world()
        self.draw_robot()

    # This function draws the world and obstacles in rviz
    def draw_world(self):
        world_quat = quaternion_from_euler(0, 0, 0)

        # Sends transformations and publishes Rviz graphical objects
        self.world_tf.sendTransform((0,0,0),world_quat,rospy.Time.now(),"map_frame", "world")
        self.world_walls_publisher.publish(self.world_walls)
        self.round_obstacles_publisher.publish(self.round_obstacles)

        # Goal updated position
        self.goal_position.pose.position.x = self.world.goal.x
        self.goal_position.pose.position.y = self.world.goal.y

        # Publishes goal
        self.goal_marker_publisher.publish(self.goal_position)

    # This function draws the robot in rviz
    def draw_robot(self):
        robot_marker_position_x = self.world.robot.x
        robot_marker_position_y = self.world.robot.y
        robot_quat = quaternion_from_euler(0, 0, self.world.robot.theta)

        # Sends transformation
        self.world_tf.sendTransform((robot_marker_position_x, robot_marker_position_y,0), robot_quat, rospy.Time.now(), "robot_frame", "world")

        # Publishes Rviz graphical objects
        self.robot_marker_publisher.publish(self.robot_position)
        self.angle_marker_publisher.publish(self.robot_angle)

if __name__ == '__main__':
    try:
        # Initialize world
        sim_dt = 0.1
        sim_freq = 1 / sim_dt
        world = World(dt = sim_dt, width = 4, height = 4)

        # Add obstacles
        world.add_obstacle(1.0, 1.0)
        world.add_obstacle(1.0, 0.0)
        world.add_obstacle(1.0, -1.0)
        world.add_obstacle(0.0, 1.0)
        world.add_obstacle(0.0, 0.0)
        world.add_obstacle(0.0, -1.0)
        world.add_obstacle(-1.0, 1.0)
        world.add_obstacle(-1.0, 0.0)
        world.add_obstacle(-1.0, -1.0)

        # Initialize node
        rospy.init_node('graphics_broadcaster', anonymous = True)
        rospy.loginfo("Starting robot simulation...")
        
        # Initialize simulation object
        turtleSimulation = RvizSimulation(world)
        turtleSimulation.draw_world() # initial drawing

        # Main ROS loop
        sim_rate = 10
        rate = rospy.Rate(sim_rate)
        while not rospy.is_shutdown():
            turtleSimulation.draw_world()
            turtleSimulation.draw_robot()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
