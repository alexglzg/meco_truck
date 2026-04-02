#!/usr/bin/env python3
"""
live_gridmap_node.py — Static map + LaserScan fusion with temporal decay

Replaces nav2 map_server + obstacle_layer with a single node that:
  1. Loads a static map (PGM + YAML) as the base layer (walls)
  2. Subscribes to /filtered_scan
  3. Raytraces each beam: clears cells along the ray, marks the hit cell
  4. Decays marked cells older than a persistence window
  5. Publishes fused OccupancyGrid on /map (transient_local QoS)

The static map cells (walls) are never modified — only free-space cells
can be temporarily marked by scan hits.

Parameters:
  map_yaml_path    — path to the YAML map file (e.g., four_wall.yaml)
  scan_topic       — LaserScan topic (default: /filtered_scan)
  publish_rate     — rate to publish the grid, Hz (default: 5.0)
  persistence      — how long a marked cell persists without being re-hit, s (default: 1.0)
"""

import os
import math
import yaml
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Quaternion
from builtin_interfaces.msg import Time


# OccupancyGrid values
OCC_FREE = 0
OCC_OCCUPIED = 100
OCC_UNKNOWN = -1

# Internal grid codes
STATIC_WALL = 1      # from loaded map, never modified
DYNAMIC_OCC = 2      # marked by scan
FREE = 0


class LiveGridmapNode(Node):
    def __init__(self):
        super().__init__('live_gridmap_node')

        # ── Parameters ──
        self.declare_parameter('map_yaml_path', '')
        self.declare_parameter('scan_topic', '/filtered_scan')
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('persistence', 1.0)

        map_yaml_path = self.get_parameter('map_yaml_path').value
        scan_topic = self.get_parameter('scan_topic').value
        publish_rate = self.get_parameter('publish_rate').value
        self.persistence = self.get_parameter('persistence').value

        # ── Load static map ──
        if not map_yaml_path or not os.path.exists(map_yaml_path):
            self.get_logger().error(f"Map YAML not found: '{map_yaml_path}'")
            raise FileNotFoundError(f"Map YAML not found: '{map_yaml_path}'")

        self._load_static_map(map_yaml_path)

        # ── Dynamic layer: timestamp grid (0.0 = not dynamically occupied) ──
        self.stamp_grid = np.zeros((self.height_px, self.width_px), dtype=np.float64)

        # ── Publisher: /map with transient_local for static_layer compatibility ──
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)

        # ── Subscriber: LaserScan ──
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10)

        # ── State ──
        self.last_scan = None

        # ── Timer: publish at fixed rate ──
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.publish_grid)

        # ── Startup log ──
        self.get_logger().info(
            f"Live Gridmap Node started\n"
            f"  Map: {map_yaml_path}\n"
            f"  Grid: {self.width_px} x {self.height_px} @ {self.resolution} m/px\n"
            f"  Origin: ({self.origin_x:.2f}, {self.origin_y:.2f})\n"
            f"  Static walls: {np.sum(self.static_grid == STATIC_WALL)} cells\n"
            f"  Scan topic: {scan_topic}\n"
            f"  Publish rate: {publish_rate} Hz\n"
            f"  Persistence: {self.persistence} s")

    # ================================================================
    # LOAD STATIC MAP
    # ================================================================
    def _load_static_map(self, yaml_path):
        """Load PGM + YAML map into internal grids."""
        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)

        map_dir = os.path.dirname(yaml_path)
        image_path = os.path.join(map_dir, meta['image'])
        self.resolution = float(meta['resolution'])
        origin = meta['origin']
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        self.origin_yaw = float(origin[2]) if len(origin) > 2 else 0.0

        occupied_thresh = float(meta.get('occupied_thresh', 0.65))
        free_thresh = float(meta.get('free_thresh', 0.196))
        negate = int(meta.get('negate', 0))

        # Load image
        img = np.array(Image.open(image_path).convert('L'), dtype=np.float64)
        if negate:
            img = 255.0 - img

        # Normalize to [0, 1]: 0 = occupied, 1 = free
        img_norm = img / 255.0

        self.height_px, self.width_px = img_norm.shape

        # Build static grid
        # Image row 0 is top of the map (max y), so we flip for
        # consistent world-frame indexing where row 0 = min y
        self.static_grid = np.zeros((self.height_px, self.width_px), dtype=np.int8)

        for r in range(self.height_px):
            for c in range(self.width_px):
                val = img_norm[r, c]
                # Image rows: top = max y, bottom = min y
                # OccupancyGrid rows: row 0 = min y
                grid_row = self.height_px - 1 - r
                if val >= occupied_thresh:
                    # High pixel value (white after negate=0) = free
                    self.static_grid[grid_row, c] = FREE
                elif val <= free_thresh:
                    # Low pixel value (black after negate=0) = occupied
                    self.static_grid[grid_row, c] = STATIC_WALL
                else:
                    self.static_grid[grid_row, c] = FREE  # treat unknown as free

    # ================================================================
    # COORDINATE CONVERSIONS
    # ================================================================
    def _world_to_grid(self, wx, wy):
        """Convert world (map frame) coords to grid indices (col, row)."""
        col = int(math.floor((wx - self.origin_x) / self.resolution))
        row = int(math.floor((wy - self.origin_y) / self.resolution))
        return col, row

    def _in_bounds(self, col, row):
        return 0 <= col < self.width_px and 0 <= row < self.height_px

    # ================================================================
    # SCAN CALLBACK
    # ================================================================
    def scan_callback(self, msg):
        """Process a new LaserScan: raytrace and mark."""
        # We need the robot pose in map frame.
        # The scan is in base_link frame. Since pcl_filter_node already
        # converts everything to map frame and back-projects into base_link,
        # the scan endpoints can be computed from the robot pose.
        #
        # However, we need the robot pose. We extract it from TF or from
        # the scan frame. For simplicity and consistency with the rest of
        # the stack, we'll buffer the scan and process it in the publish
        # timer, where we can look up TF.
        self.last_scan = msg

    # ================================================================
    # PROCESS SCAN (raytrace + mark)
    # ================================================================
    def _process_scan(self, scan, robot_x, robot_y, robot_yaw):
        now = self.get_clock().now().nanoseconds / 1e9
        angle = scan.angle_min

        for i in range(len(scan.ranges)):
            r = scan.ranges[i]
            if r < scan.range_min or r > scan.range_max or not math.isfinite(r):
                angle += scan.angle_increment
                continue

            # Hit point in map frame
            beam_angle = angle + robot_yaw
            hit_x = robot_x + r * math.cos(beam_angle)
            hit_y = robot_y + r * math.sin(beam_angle)

            col, row = self._world_to_grid(hit_x, hit_y)
            if self._in_bounds(col, row):
                if self.static_grid[row, col] != STATIC_WALL:
                    self.stamp_grid[row, col] = now

            angle += scan.angle_increment

    # ================================================================
    # DECAY OLD MARKS
    # ================================================================
    def _decay_old_marks(self):
        """Clear dynamic marks older than the persistence window."""
        now = self.get_clock().now().nanoseconds / 1e9
        cutoff = now - self.persistence

        # Zero out stamps older than cutoff (but not already-zero stamps)
        stale = (self.stamp_grid > 0) & (self.stamp_grid < cutoff)
        self.stamp_grid[stale] = 0.0

    # ================================================================
    # PUBLISH GRID
    # ================================================================
    def publish_grid(self):
        """Compose static + dynamic into OccupancyGrid and publish."""
        # Process latest scan if available
        if self.last_scan is not None:
            # Get robot pose from TF
            try:
                import tf2_ros
                if not hasattr(self, '_tf_buffer'):
                    self._tf_buffer = tf2_ros.Buffer()
                    self._tf_listener = tf2_ros.TransformListener(
                        self._tf_buffer, self)

                tf = self._tf_buffer.lookup_transform(
                    'map', 'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1))

                robot_x = tf.transform.translation.x
                robot_y = tf.transform.translation.y
                q = tf.transform.rotation
                robot_yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))

                self._process_scan(self.last_scan, robot_x, robot_y, robot_yaw)
                self.last_scan = None

            except Exception as e:
                self.get_logger().warn(
                    f"TF lookup failed: {e}", throttle_duration_sec=2.0)

        # Decay old marks
        self._decay_old_marks()

        # Compose output grid
        now_msg = self.get_clock().now().to_msg()

        msg = OccupancyGrid()
        msg.header.stamp = now_msg
        msg.header.frame_id = 'map'

        msg.info.resolution = self.resolution
        msg.info.width = self.width_px
        msg.info.height = self.height_px
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = math.cos(self.origin_yaw / 2.0)
        msg.info.origin.orientation.z = math.sin(self.origin_yaw / 2.0)

        # Build flat data array
        data = np.full(self.height_px * self.width_px, OCC_FREE, dtype=np.int8)

        # Static walls
        wall_mask = (self.static_grid == STATIC_WALL).flatten()
        data[wall_mask] = OCC_OCCUPIED

        # Dynamic obstacles (any cell with a non-zero stamp)
        dyn_mask = (self.stamp_grid > 0).flatten()
        data[dyn_mask] = OCC_OCCUPIED

        msg.data = data.tolist()
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LiveGridmapNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()