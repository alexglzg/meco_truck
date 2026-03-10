#!/usr/bin/env python3
"""
Goal-to-Plan Bridge Node (ROS2 Foxy)

Bridges RViz2's "2D Nav Goal" and nav2's planner server.

Jobs:
  1. Broadcasts map->base_link TF from /truck/state so NavFn can
     find the robot's start pose (and the costmap stops complaining).
  2. On /goal_pose, calls ComputePathToPose action on the planner server.
  3. Replans at a fixed frequency (like move_base's planner_frequency).
  4. Publishes the result on /plan for the MPC-CBF and FIRI nodes.
  5. Stops replanning when the robot reaches the goal.

Foxy-specific:
  - ComputePathToPose.Goal uses 'pose' (not 'goal')
  - No 'start' or 'use_start' fields -- NavFn always uses TF for start
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

        # -- Parameters --
        self.declare_parameter('replan_frequency', 1.0)    # Hz, 0 = plan once
        self.declare_parameter('goal_tolerance', 0.15)     # meters
        self.declare_parameter('planner_id', 'GridBased')

        self.replan_freq = self.get_parameter('replan_frequency').value
        self.goal_tol = self.get_parameter('goal_tolerance').value
        self.planner_id = self.get_parameter('planner_id').value

        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.state_received = False

        # Active goal (None = no goal)
        self.active_goal = None          # PoseStamped
        self.planning_in_progress = False

        # TF broadcaster (map -> base_link)
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

        # -- Replan timer --
        if self.replan_freq > 0.0:
            period_s = 1.0 / self.replan_freq
            self.replan_timer = self.create_timer(
                period_s, self.replan_callback)
            self.get_logger().info(
                f"Replanning at {self.replan_freq:.1f} Hz "
                f"(every {period_s*1000:.0f} ms)")
        else:
            self.replan_timer = None
            self.get_logger().info("Replanning disabled (plan once per goal)")

        self.get_logger().info(
            f"Goal-to-Plan bridge ready. "
            f"goal_tolerance={self.goal_tol:.2f} m, "
            f"planner_id='{self.planner_id}'. "
            f"Click '2D Nav Goal' in RViz2 to trigger NavFn.")

    def state_callback(self, msg):
        """Track robot state and broadcast TF."""
        if len(msg.data) < 3:
            return

        self.robot_x = msg.data[0]
        self.robot_y = msg.data[1]
        self.robot_theta = msg.data[2]
        self.state_received = True

        # Broadcast map -> base_link transform
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
        """Receive goal from RViz2. Stores it and plans immediately."""

        self.get_logger().info(
            f"New goal: [{msg.pose.position.x:.2f}, "
            f"{msg.pose.position.y:.2f}]")

        self.active_goal = msg

        # Plan immediately (don't wait for next timer tick)
        self.request_plan()

    def replan_callback(self):
        """Periodic replanning timer. Fires at replan_frequency."""

        if self.active_goal is None:
            return

        if not self.state_received:
            return

        # Check if we've reached the goal
        dx = self.active_goal.pose.position.x - self.robot_x
        dy = self.active_goal.pose.position.y - self.robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.goal_tol:
            self.get_logger().info(
                f"Goal reached (d={dist:.3f} m < tol={self.goal_tol:.2f} m). "
                f"Stopping replanning.")
            self.active_goal = None
            return

        # Don't stack requests if one is in flight
        if self.planning_in_progress:
            return

        self.request_plan()

    def request_plan(self):
        """Send a ComputePathToPose request to the planner server."""

        if not self.state_received:
            self.get_logger().warn("No robot state yet, can't plan")
            return

        if self.active_goal is None:
            return

        if not self._action_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn(
                "Planner server not available, is it running?")
            return

        # Build the action goal (Foxy field names)
        goal = ComputePathToPose.Goal()
        goal.pose = self.active_goal         # Foxy: 'pose', not 'goal'
        goal.planner_id = self.planner_id

        self.planning_in_progress = True

        # Send goal asynchronously
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        """Handle the action server's acceptance/rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Plan request rejected by planner server")
            self.planning_in_progress = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        """Publish the computed path on /plan."""
        self.planning_in_progress = False

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