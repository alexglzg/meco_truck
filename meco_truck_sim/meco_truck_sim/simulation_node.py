#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import numpy as np
from numpy import sin, cos, tan
import math

import sys
import os
sys.path.append(os.path.dirname(__file__))  # adds current directory


class TruckSimulator(Node):
    def __init__(self):
        super().__init__('truck_simulator')
        self.get_logger().info('Truck Simulator Started')

        # System parameters
        self.L0 = 0.42   # truck wheelbase

        # State: [x0, y0, theta0]
        # x0, y0: truck rear axle position
        # theta0: truck heading
        self.state = np.array([0.0, 0.0, 0.0])

        # Control inputs
        self.V0 = 0.0      # truck velocity
        self.delta0 = 0.0  # steering angle

        # Publishers
        self.state_pub = self.create_publisher(
            Float64MultiArray,
            '/truck/state',
            10
        )

        self.truck_pose_pub = self.create_publisher(
            PoseStamped,
            '/truck/pose',
            10
        )

        # Odometry for RViz
        self.odom_pub = self.create_publisher(
            Odometry,
            '/odom',
            10
        )

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Control input subscriber
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        # Subscriber to set initial pose from Rviz2 2D Pose Estimate
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initialpose_callback,
            10
        )

        # Simulation timer (20 Hz)
        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.simulation_step)
        self.last_time = self.get_clock().now()


    def cmd_callback(self, msg):
        """Receive control commands"""
        # Linear velocity (m/s)
        self.V0 = max(-0.5, min(0.5, msg.linear.x))

        # Steering angle (rad)
        max_steering = 180.0 * np.pi / 180.0
        self.delta0 = max(-max_steering, min(max_steering, msg.angular.z))

    def initialpose_callback(self, msg):
        """Set initial pose from Rviz2 2D Pose Estimate"""
        self.state[0] = msg.pose.pose.position.x
        self.state[1] = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.state[2] = yaw  # theta0 (truck heading)

        self.get_logger().warn(
            f'Initial pose set to x: {self.state[0]:.2f}, y: {self.state[1]:.2f}, theta: {yaw:.2f} rad'
        )

    def bicycle_dynamics(self, state, u):
        """
        Compute state derivatives using bicycle kinematic model.

        state = [x0, y0, theta0]
        u = [V0, delta0]
        """
        x0, y0, th0 = state
        V0, delta0 = u

        dx0 = V0 * cos(th0)
        dy0 = V0 * sin(th0)
        dth0 = V0 * tan(delta0) / self.L0

        return np.array([dx0, dy0, dth0])

    def euler_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        q = Quaternion()
        q.w = cy
        q.x = 0.0
        q.y = 0.0
        q.z = sy
        return q

    def simulation_step(self):
        """Main simulation loop"""
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9

        if dt <= 0:
            return

        # Integrate dynamics (Euler integration)
        u = np.array([self.V0, self.delta0])
        state_dot = self.bicycle_dynamics(self.state, u)
        self.state = self.state + self.dt * state_dot

        # Unwrap angle to [-pi, pi]
        self.state[2] = self.normalize_angle(self.state[2])  # theta0

        # Publish all topics
        self.publish_state(now)
        self.publish_transforms(now)

        self.last_time = now

    def normalize_angle(self, angle):
        """Wrap angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def publish_state(self, now):
        """Publish truck state"""
        x0, y0, th0 = self.state

        # State vector [x0, y0, theta0]
        state_msg = Float64MultiArray()
        state_msg.data = [float(x0), float(y0), float(th0)]
        self.state_pub.publish(state_msg)

        # Truck pose (rear axle is origin)
        truck_msg = PoseStamped()
        truck_msg.header.stamp = now.to_msg()
        truck_msg.header.frame_id = 'map'
        truck_msg.pose.position.x = float(x0)
        truck_msg.pose.position.y = float(y0)
        truck_msg.pose.position.z = 0.0
        truck_msg.pose.orientation = self.euler_to_quaternion(th0)
        self.truck_pose_pub.publish(truck_msg)

        # Odometry (for RViz)
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = float(x0)
        odom.pose.pose.position.y = float(y0)
        odom.pose.pose.orientation = self.euler_to_quaternion(th0)
        odom.twist.twist.linear.x = float(self.V0)
        odom.twist.twist.angular.z = float(self.delta0)
        self.odom_pub.publish(odom)

        # Joint states for front wheel steering
        js = JointState()
        js.header.stamp = now.to_msg()
        js.name = ['front_wheel_joint']
        js.position = [float(self.delta0)]
        self.joint_pub.publish(js)

    def publish_transforms(self, now):
        """Publish TF transforms"""
        x0, y0, th0 = self.state

        # Map -> base_link (truck rear axle)
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(x0)
        t.transform.translation.y = float(y0)
        t.transform.translation.z = 0.0
        t.transform.rotation = self.euler_to_quaternion(th0)

        self.tf_broadcaster.sendTransform(t)

    def quaternion_to_yaw(self, q):
        """
        Convert quaternion (x,y,z,w) to yaw in radians.
        """
        x, y, z, w = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = TruckSimulator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
