import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Float32
from tf2_ros import TransformBroadcaster
import math
import numpy as np

class TruckEstimator(Node):
    def __init__(self):
        super().__init__('truck_estimator')

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.state_pub = self.create_publisher(Float64MultiArray, '/truck/state', 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_received = False

    def quaternion_to_yaw(self, q):
        # Convert quaternion to yaw angle
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(q)

        self.odom_received = True
        self.publish_full_state(msg.header.stamp)

    def publish_full_state(self, stamp):
        if not self.odom_received:
            return

        state_msg = Float64MultiArray()
        state_msg.data = [self.x, self.y, self.yaw]
        self.state_pub.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TruckEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()