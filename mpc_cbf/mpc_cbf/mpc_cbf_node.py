#!/usr/bin/env python3
"""
MPC-CBF Node for Bicycle-Model AMR (ROS2 Foxy)

Combines:
  - Smooth min/max polytope-in-polytope CBF (from ROS1 vessel mpc_cbf_path_node)
  - opti.solve() with manual warm start (robust with Fatrop)
  - Bicycle kinematics (from truck simulation_node)
  - firi_msgs/Polytope2DStamped input (from ROS2 FIRI node)

Reference priority:
  1. Path following (nav2 /plan or any Path topic) with lookahead
  2. RViz2 2D Nav Goal (/goal_pose)
  3. Hold current position (fallback)

All computation in ENU/map frame. No NED transform.

Topics:
  Subscribed:
    /truck/state     (Float64MultiArray)    — [x, y, theta] from sim
    /firi/polytope   (Polytope2DStamped)    — from FIRI node
    /goal_pose       (PoseStamped)          — RViz2 2D Nav Goal
    /plan            (Path)                 — nav2 NavFn global planner
  Published:
    /cmd_vel         (Twist)                — velocity + steering to sim
    /mpc/trajectory  (Path)                 — predicted trajectory for RViz2
    /mpc/reference   (PoseStamped)          — current reference point
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray, String
from nav_msgs.msg import Path
from firi_msgs.msg import Polytope2DStamped

import casadi as ca
import numpy as np
import math


class MPCCBFNode(Node):
    def __init__(self):
        super().__init__('mpc_cbf_node')

        # =================================================================
        # Parameters
        # =================================================================
        # Vehicle
        self.declare_parameter('L0', 0.42)
        self.declare_parameter('robot_length', 0.555)
        self.declare_parameter('robot_width', 0.35)
        self.declare_parameter('footprint_offset_x', 0.1975)
        self.declare_parameter('max_velocity', 0.5)
        self.declare_parameter('max_steering_deg', 50.0)

        # Horizon
        self.declare_parameter('horizon', 15)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('N_cbf', 6)
        self.declare_parameter('rate', 10.0)

        # CBF
        self.declare_parameter('gamma', 0.8)
        self.declare_parameter('max_approx', 5e-3)
        self.declare_parameter('max_halfplanes', 20)

        # Cost weights
        self.declare_parameter('Q_x', 10.0)
        self.declare_parameter('Q_y', 10.0)
        self.declare_parameter('Q_theta', 0.0)
        self.declare_parameter('R_v', 0.1)
        self.declare_parameter('R_delta', 0.01)
        self.declare_parameter('R_smooth_v', 0.05)
        self.declare_parameter('R_smooth_delta', 0.05)

        # Path following
        self.declare_parameter('lookahead_dist', 0.5)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('path_timeout', 5.0)

        # Read parameters
        self.L0 = self.get_parameter('L0').value
        self.robot_length = self.get_parameter('robot_length').value
        self.robot_width = self.get_parameter('robot_width').value
        self.footprint_offset_x = self.get_parameter('footprint_offset_x').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_steering = np.radians(self.get_parameter('max_steering_deg').value)

        self.N = self.get_parameter('horizon').value
        self.dt = self.get_parameter('dt').value
        self.N_cbf = self.get_parameter('N_cbf').value
        self.rate = self.get_parameter('rate').value

        self.gamma = self.get_parameter('gamma').value
        self.max_approx = self.get_parameter('max_approx').value
        self.max_halfplanes = self.get_parameter('max_halfplanes').value

        Q_x = self.get_parameter('Q_x').value
        Q_y = self.get_parameter('Q_y').value
        Q_theta = self.get_parameter('Q_theta').value
        R_v = self.get_parameter('R_v').value
        R_delta = self.get_parameter('R_delta').value
        R_smooth_v = self.get_parameter('R_smooth_v').value
        R_smooth_delta = self.get_parameter('R_smooth_delta').value

        self.Q = np.diag([Q_x, Q_y, Q_theta])
        self.R = np.diag([R_v, R_delta])
        self.R_smooth = np.diag([R_smooth_v, R_smooth_delta])
        self.Q_terminal = self.Q * 10

        self.lookahead_dist = self.get_parameter('lookahead_dist').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.path_timeout = self.get_parameter('path_timeout').value

        # State dimensions
        self.nx = 3   # [x, y, theta]
        self.nu = 2   # [v, delta]

        # =================================================================
        # State variables
        # =================================================================
        self.current_state = np.zeros(self.nx)
        self.state_received = False
        self.u_prev = np.zeros(self.nu)

        # Reference (ENU frame)
        self.x_ref = np.zeros(self.nx)
        self.ref_received = False
        self._active_source = "none"

        # Polytope (fixed-size parameterization)
        self.A_poly = np.zeros((self.max_halfplanes, 2))
        self.b_poly = np.ones(self.max_halfplanes) * 10.0
        self.n_active = 0
        self.poly_received = False

        # Path following state
        self._path = None
        self._path_stamp = None
        self._rviz_goal = None
        self._hold_pos = None

        # Warm start (previous solution)
        self.init_guess_x = None
        self.init_guess_u = None

        # =================================================================
        # Build OCP
        # =================================================================
        self.get_logger().info("Compiling MPC-CBF...")
        self._build_ocp()
        self.get_logger().info("MPC-CBF compiled and ready.")

        # =================================================================
        # Publishers
        # =================================================================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.traj_pub = self.create_publisher(Path, '/mpc/trajectory', 10)
        self.ref_pub = self.create_publisher(PoseStamped, '/mpc/reference', 10)
        self.status_pub = self.create_publisher(String, '/mpc/status', 10)

        # =================================================================
        # Subscribers
        # =================================================================
        self.create_subscription(
            Float64MultiArray, '/truck/state',
            self.state_callback, 10)

        self.create_subscription(
            Polytope2DStamped, '/firi/polytope',
            self.poly_callback, 10)

        self.create_subscription(
            PoseStamped, '/goal_pose',
            self.goal_callback, 10)

        self.create_subscription(
            Path, '/plan',
            self.path_callback, 10)

        # =================================================================
        # Timer: control loop
        # =================================================================
        timer_period = 1.0 / self.rate
        self.timer = self.create_timer(timer_period, self.control_loop)

        # Startup log
        self.get_logger().info(
            f"MPC-CBF Node started\n"
            f"  Vehicle: L0={self.L0}, {self.robot_length}x{self.robot_width}m, "
            f"offset={self.footprint_offset_x}m\n"
            f"  Horizon: N={self.N}, dt={self.dt}, N_cbf={self.N_cbf}\n"
            f"  CBF: gamma={self.gamma}, max_approx={self.max_approx}, "
            f"max_hp={self.max_halfplanes}\n"
            f"  Limits: v_max={self.max_velocity}, "
            f"delta_max={np.degrees(self.max_steering):.1f}deg\n"
            f"  Ref priority: path > rviz_goal > hold")

    # =====================================================================
    # Dynamics
    # =====================================================================
    def _bicycle_dynamics(self, x, u):
        """
        Bicycle kinematic model (CasADi symbolic).

        State:  x = [x, y, theta]
        Input:  u = [v, delta]   (velocity, steering angle)

        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = v * tan(delta) / L0
        """
        theta = x[2]
        v = u[0]
        delta = u[1]

        return ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            v * ca.tan(delta) / self.L0
        )

    def _robot_vertices_ca(self, x, y, theta):
        """
        Compute 4 robot footprint vertices in world frame (CasADi symbolic).

        Accounts for the footprint offset: body center is
        footprint_offset_x ahead of base_link (rear axle).

        Returns a 4x2 CasADi matrix: each row is [vx, vy].
        """
        hl = self.robot_length / 2.0
        hw = self.robot_width / 2.0
        ox = self.footprint_offset_x

        c = ca.cos(theta)
        s = ca.sin(theta)

        # Body center in world frame
        cx = x + ox * c
        cy = y + ox * s

        # Local corners relative to body center
        local_x = ca.vertcat(hl, hl, -hl, -hl)
        local_y = ca.vertcat(hw, -hw, -hw, hw)

        # Rotate and translate
        world_x = c * local_x - s * local_y + cx
        world_y = s * local_x + c * local_y + cy

        return ca.horzcat(world_x, world_y)

    # =====================================================================
    # OCP Construction
    # =====================================================================
    def _build_ocp(self):
        """
        Build the CasADi OCP with:
          - Bicycle dynamics
          - Fixed-size polytope CBF (smooth min/max, inlined)
          - Input constraints + smoothing
          - opti.solve() with manual warm start (robust with Fatrop)

        The smooth_min is inlined (not wrapped in ca.Function) so Fatrop
        can trace variable dependencies and assign each CBF constraint
        to the correct OCP stage. Using ca.Function creates a "black box"
        that breaks Fatrop's structure detection.
        """
        self.opti = ca.Opti()

        # ── Decision variables ──
        self.X = []
        self.U = []

        for k in range(self.N):
            self.X.append(self.opti.variable(self.nx))
            self.U.append(self.opti.variable(self.nu))
        self.X.append(self.opti.variable(self.nx))  # terminal

        # ── Parameters ──
        self.p_X0 = self.opti.parameter(self.nx)
        self.p_Ref = self.opti.parameter(self.nx)
        self.p_U_prev = self.opti.parameter(self.nu)
        self.p_A = self.opti.parameter(self.max_halfplanes, 2)
        self.p_b = self.opti.parameter(self.max_halfplanes)

        # ── Smooth min (inlined, no ca.Function wrapper) ──
        n_total = 4 * self.max_halfplanes
        alpha_val = ca.log(n_total) / self.max_approx

        def smooth_min_inline(e_vec, alpha):
            """smooth_min(e) = -smooth_max(-e) = -(1/a)*logsumexp(-a*e)"""
            ae = -alpha * e_vec
            m = ca.mmax(ae)
            return -(1.0 / alpha) * (m + ca.log(ca.sum1(ca.exp(ae - m))))

        # ── Cost + Dynamics + CBF ──
        cost = 0

        for k in range(self.N):
            xk = self.X[k]
            uk = self.U[k]
            x_next = self.X[k + 1]

            # ── Cost ──
            err = xk - self.p_Ref
            cost += ca.mtimes([err.T, self.Q, err])
            cost += ca.mtimes([uk.T, self.R, uk])

            # Input smoothing
            if k == 0:
                du = uk - self.p_U_prev
            else:
                du = uk - self.U[k - 1]
            cost += ca.mtimes([du.T, self.R_smooth, du])

            # ── Dynamics (Euler integration) ──
            dxdt = self._bicycle_dynamics(xk, uk)
            x_next_model = xk + dxdt * self.dt
            self.opti.subject_to(x_next == x_next_model)

            # ── Initial state ──
            if k == 0:
                self.opti.subject_to(self.X[0] == self.p_X0)

            # ── Input bounds ──
            self.opti.subject_to(self.opti.bounded(
                -self.max_velocity, uk[0], self.max_velocity))
            self.opti.subject_to(self.opti.bounded(
                -self.max_steering, uk[1], self.max_steering))

            # ── CBF constraint (at this stage) ──
            if k < self.N_cbf:
                # Slacks at X[k]
                verts_k = self._robot_vertices_ca(xk[0], xk[1], xk[2])
                slacks_k = []
                for j in range(4):
                    for i in range(self.max_halfplanes):
                        slacks_k.append(
                            self.p_b[i] - (self.p_A[i, 0] * verts_k[j, 0]
                                         + self.p_A[i, 1] * verts_k[j, 1]))

                h_k = smooth_min_inline(ca.vertcat(*slacks_k), alpha_val)

                # Slacks at model prediction (depends on xk, uk only)
                verts_k1 = self._robot_vertices_ca(
                    x_next_model[0], x_next_model[1], x_next_model[2])
                slacks_k1 = []
                for j in range(4):
                    for i in range(self.max_halfplanes):
                        slacks_k1.append(
                            self.p_b[i] - (self.p_A[i, 0] * verts_k1[j, 0]
                                         + self.p_A[i, 1] * verts_k1[j, 1]))

                h_k1 = smooth_min_inline(ca.vertcat(*slacks_k1), alpha_val)

                # CBF decrease condition
                self.opti.subject_to(
                    h_k1 >= self.gamma * h_k
                           + (1 - self.gamma) * self.max_approx)

        # Terminal cost
        err_t = self.X[self.N] - self.p_Ref
        cost += ca.mtimes([err_t.T, self.Q_terminal, err_t])

        self.opti.minimize(cost)

        # ── Solver options ──
        opts = {
            "fatrop.print_level": 0,
            "print_time": 0,
            "fatrop.max_iter": 100,
            "fatrop.tol": 1e-4,
            "fatrop.mu_init": 1e-1,
            "structure_detection": "auto",
            "expand": True,
            "debug": False
        }
        self.opti.solver('fatrop', opts)

        self.get_logger().info(
            f"  OCP built: {self.N} steps, {self.N_cbf} CBF steps, "
            f"{self.max_halfplanes} max halfplanes")

    # =====================================================================
    # Callbacks
    # =====================================================================
    def state_callback(self, msg):
        """Receive robot state [x, y, theta] from simulator."""
        if len(msg.data) < 3:
            return
        self.current_state[0] = msg.data[0]
        self.current_state[1] = msg.data[1]
        self.current_state[2] = msg.data[2]
        self.state_received = True

    def poly_callback(self, msg):
        """
        Receive polytope from FIRI node.
        Unpack into fixed-size A_poly, b_poly arrays.
        Unused slots filled with dummy constraints (0*x <= 10).
        """
        self.A_poly = np.zeros((self.max_halfplanes, 2))
        self.b_poly = np.ones(self.max_halfplanes) * 10.0

        n = min(len(msg.polytope.planes), self.max_halfplanes)
        for i in range(n):
            self.A_poly[i, 0] = msg.polytope.planes[i].normal[0]
            self.A_poly[i, 1] = msg.polytope.planes[i].normal[1]
            self.b_poly[i] = msg.polytope.planes[i].offset

        self.n_active = n
        self.poly_received = True

    def goal_callback(self, msg):
        """RViz2 2D Nav Goal. Store as [x, y, yaw] in ENU."""
        x = msg.pose.position.x
        y = msg.pose.position.y

        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        self._rviz_goal = np.array([x, y, yaw])
        self.get_logger().info(
            f"RViz goal: [{x:.2f}, {y:.2f}, {np.degrees(yaw):.1f}deg]")

    def path_callback(self, msg):
        """Path from nav2 global planner. Store in ENU."""
        if len(msg.poses) < 2:
            return
        self._path = msg
        self._path_stamp = self.get_clock().now()

    # =====================================================================
    # Path following: lookahead reference extraction
    # =====================================================================
    def _extract_path_reference(self):
        """
        Extract reference [x, y, yaw] from stored path using lookahead.

        1. Find closest point on path to robot.
        2. Walk forward by lookahead_dist.
        3. Compute yaw from path tangent at lookahead point.

        Returns None if path is exhausted (robot near end).
        """
        if self._path is None or len(self._path.poses) < 2:
            return None

        poses = self._path.poses
        rx, ry = self.current_state[0], self.current_state[1]

        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0
        for i, p in enumerate(poses):
            dx = p.pose.position.x - rx
            dy = p.pose.position.y - ry
            d = dx * dx + dy * dy
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Walk forward by lookahead distance
        remaining = self.lookahead_dist

        for i in range(closest_idx, len(poses) - 1):
            dx = poses[i + 1].pose.position.x - poses[i].pose.position.x
            dy = poses[i + 1].pose.position.y - poses[i].pose.position.y
            seg_len = math.sqrt(dx * dx + dy * dy)

            if seg_len < 1e-6:
                continue

            if remaining <= seg_len:
                frac = remaining / seg_len
                ref_x = poses[i].pose.position.x + frac * dx
                ref_y = poses[i].pose.position.y + frac * dy
                ref_yaw = math.atan2(dy, dx)
                return np.array([ref_x, ref_y, ref_yaw])

            remaining -= seg_len

        # Reached end of path
        last = poses[-1]
        ref_x = last.pose.position.x
        ref_y = last.pose.position.y

        if len(poses) >= 2:
            prev = poses[-2]
            dx = last.pose.position.x - prev.pose.position.x
            dy = last.pose.position.y - prev.pose.position.y
            if dx * dx + dy * dy > 1e-6:
                ref_yaw = math.atan2(dy, dx)
            else:
                ref_yaw = self.current_state[2]
        else:
            ref_yaw = self.current_state[2]

        # Check if path is exhausted
        dx = ref_x - rx
        dy = ref_y - ry
        if math.sqrt(dx * dx + dy * dy) < self.goal_tolerance:
            return None

        return np.array([ref_x, ref_y, ref_yaw])

    # =====================================================================
    # Reference manager: priority-based selection
    # =====================================================================
    def _update_reference(self):
        """
        Select active reference by priority:
          1. Path (from nav2 /plan) — if recent and not exhausted
          2. RViz goal (/goal_pose) — persistent until reached
          3. Hold current position — fallback
        """
        ref = None

        # ── Priority 1: Path following ──
        if self._path is not None and self._path_stamp is not None:
            age = (self.get_clock().now() - self._path_stamp).nanoseconds / 1e9
            if age < self.path_timeout:
                ref = self._extract_path_reference()
                if ref is not None:
                    self._active_source = "path"
                else:
                    self._path = None
                    self._path_stamp = None

        # ── Priority 2: RViz goal ──
        if ref is None and self._rviz_goal is not None:
            dx = self._rviz_goal[0] - self.current_state[0]
            dy = self._rviz_goal[1] - self.current_state[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > self.goal_tolerance:
                ref = self._rviz_goal
                self._active_source = "rviz"
            else:
                ref = self._rviz_goal
                self._active_source = "rviz_hold"

        # ── Priority 3: Hold position ──
        if ref is None:
            if self._hold_pos is None and self.state_received:
                self._hold_pos = self.current_state.copy()
                self.get_logger().info("No reference, holding position")

            if self._hold_pos is not None:
                self.x_ref = self._hold_pos
                self._active_source = "hold"
                self.ref_received = True
                return

            self._active_source = "none"
            return

        # ── Set reference ──
        self.x_ref = ref
        self.ref_received = True
        self._hold_pos = None

    # =====================================================================
    # Control loop
    # =====================================================================
    def control_loop(self):
        """Timer callback: update reference, solve MPC, publish."""

        if not self.state_received:
            self.get_logger().warn("Waiting for state...",
                                   throttle_duration_sec=2.0)
            return

        self._update_reference()
        if not self.ref_received:
            self.get_logger().warn("Waiting for reference...",
                                   throttle_duration_sec=2.0)
            return

        # ── Set parameter values ──
        self.opti.set_value(self.p_X0, self.current_state)
        self.opti.set_value(self.p_Ref, self.x_ref)
        self.opti.set_value(self.p_U_prev, self.u_prev)
        self.opti.set_value(self.p_A, self.A_poly)
        self.opti.set_value(self.p_b, self.b_poly)

        # ── Warm start from previous solution ──
        if self.init_guess_x is not None:
            for k in range(self.N):
                self.opti.set_initial(self.X[k], self.init_guess_x[:, k])
                self.opti.set_initial(self.U[k], self.init_guess_u[:, k])
            self.opti.set_initial(
                self.X[self.N], self.init_guess_x[:, self.N])

        # ── Solve ──
        try:
            sol = self.opti.solve()

            # Extract solution
            u_opt = np.zeros((self.nu, self.N))
            x_opt = np.zeros((self.nx, self.N + 1))

            for k in range(self.N):
                u_opt[:, k] = sol.value(self.U[k])
                x_opt[:, k] = sol.value(self.X[k])
            x_opt[:, self.N] = sol.value(self.X[self.N])

            # Save for warm start
            self.init_guess_x = x_opt
            self.init_guess_u = u_opt

            # Apply first control
            self.u_prev = u_opt[:, 0]

            # Publish
            self.publish_cmd(u_opt[:, 0])
            self.publish_trajectory(x_opt)
            self.publish_reference()
            self.publish_status()

            self.get_logger().info(
                f"[{self._active_source}] v={u_opt[0, 0]:.3f} m/s, "
                f"delta={np.degrees(u_opt[1, 0]):.1f}deg, "
                f"hp={self.n_active}",
                throttle_duration_sec=0.5)

        except RuntimeError as e:
            self.get_logger().warn(f"MPC solve failed: {e}",
                                   throttle_duration_sec=1.0)
            try:
                u_debug = np.array(
                    self.opti.debug.value(self.U[0])).flatten()
                self.publish_cmd(u_debug)
            except Exception:
                self.cmd_pub.publish(Twist())

    # =====================================================================
    # Publishers
    # =====================================================================
    def publish_cmd(self, u):
        """
        Publish velocity and steering to the simulator.
        Matches simulation_node cmd_callback:
          linear.x  = velocity (m/s)
          angular.z = steering angle (rad)
        """
        msg = Twist()
        msg.linear.x = float(u[0])
        msg.angular.z = float(u[1])
        self.cmd_pub.publish(msg)

    def publish_trajectory(self, traj):
        """Publish predicted trajectory as Path for RViz2."""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for k in range(traj.shape[1]):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(traj[0, k])
            pose.pose.position.y = float(traj[1, k])
            pose.pose.position.z = 0.0

            yaw = float(traj[2, k])
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

            msg.poses.append(pose)

        self.traj_pub.publish(msg)

    def publish_reference(self):
        """Publish current reference point for RViz2."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(self.x_ref[0])
        msg.pose.position.y = float(self.x_ref[1])
        msg.pose.position.z = 0.0

        yaw = float(self.x_ref[2])
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.ref_pub.publish(msg)

    def publish_status(self):
        """Publish active reference source for debugging."""
        self.status_pub.publish(String(data=self._active_source))


def main(args=None):
    rclpy.init(args=args)
    node = MPCCBFNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()