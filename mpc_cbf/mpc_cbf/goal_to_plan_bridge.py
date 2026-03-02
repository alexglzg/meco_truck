#!/usr/bin/env python3
"""
Goal-to-Plan Bridge Node (ROS2 Foxy)

Bridges RViz2's "2D Nav Goal" and nav2's planner server.

Three jobs:
  1. Broadcasts map→base_link TF from /truck/state so NavFn can
     find the robot's start pose (and the costmap stops complaining).
  2. On /goal_pose, calls ComputePathToPose action on the planner server.
  3. Publishes the result on /plan for the MPC-CBF node.

Foxy-specific:
  - ComputePathToPose.Goal uses 'pose' (not 'goal')
  - No 'start' or 'use_start' fields — NavFn always uses TF for start
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose
from tf2_ros import TransformBroadcaster


class GoalToPlanBridge(Node):
    def __init__(self):
        super().__init__('goal_to_plan_bridge')

        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.state_received = False

        # TF broadcaster (map → base_link)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publisher for the computed path
        self.plan_pub = self.create_publisher(Path, '/plan', 10)

        # Subscribe to robot state
        self.create_subscription(
            Float64MultiArray, '/truck/state',
            self.state_callback, 10)

        # Subscribe to RViz2 goal
        self.create_subscription(
            PoseStamped, '/goal_pose',
            self.goal_callback, 10)

        # Action client for nav2 planner
        self._action_client = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose')

        self.get_logger().info(
            "Goal-to-Plan bridge ready. "
            "Click '2D Nav Goal' in RViz2 to trigger NavFn.")

    def state_callback(self, msg):
        """Track robot state and broadcast TF."""
        if len(msg.data) < 3:
            return

        self.robot_x = msg.data[0]
        self.robot_y = msg.data[1]
        self.robot_theta = msg.data[2]
        self.state_received = True

        # Broadcast map → base_link transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.robot_x
        t.transform.translation.y = self.robot_y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(self.robot_theta / 2.0)
        t.transform.rotation.w = math.cos(self.robot_theta / 2.0)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0

        self.tf_broadcaster.sendTransform(t)

    def goal_callback(self, msg):
        """Receive goal from RViz2 and request a plan from NavFn."""

        self.get_logger().info(
            f"Goal received: [{msg.pose.position.x:.2f}, "
            f"{msg.pose.position.y:.2f}]")

        if not self.state_received:
            self.get_logger().warn("No robot state yet, can't plan")
            return

        # Wait for action server
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn(
                "Planner server not available, is it running?")
            return

        # Build the action goal (Foxy field names)
        goal = ComputePathToPose.Goal()
        goal.pose = msg                    # Foxy: 'pose', not 'goal'
        goal.planner_id = "GridBased"
        # Foxy has no 'start' or 'use_start' — NavFn uses TF (our broadcast)

        self.get_logger().info(
            f"Planning from [{self.robot_x:.2f}, {self.robot_y:.2f}] "
            f"to [{msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}]")

        # Send goal asynchronously
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        """Handle the action server's acceptance/rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Plan request rejected by planner server")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        """Publish the computed path on /plan."""
        result = future.result().result
        path = result.path

        if len(path.poses) == 0:
            self.get_logger().warn("Planner returned empty path")
            return

        self.plan_pub.publish(path)
        self.get_logger().info(
            f"Plan published: {len(path.poses)} poses")


def main(args=None):
    rclpy.init(args=args)
    node = GoalToPlanBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()