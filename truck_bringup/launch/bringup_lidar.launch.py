"""
Bringup for Nav2 planner with Ouster lidar obstacle layer.

Launches:
  1. Map server (four_wall.yaml) + lifecycle manager
  2. PCL filter node (publishes /filtered_cloud and /filtered_scan)
  3. Nav2 planner server (NavFn A*) + lifecycle manager
  4. Goal-to-plan bridge (RViz2 goal → NavFn action → /plan)

Usage:
  ros2 launch truck_bringup bringup_lidar.launch.py
  ros2 launch truck_bringup bringup_lidar.launch.py map_file:=four_wall.yaml
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():

    # ── Package directories ──
    bringup_dir = get_package_share_directory('truck_bringup')
    sim_dir = get_package_share_directory('meco_truck_sim')
    firi_dir = get_package_share_directory('firi_ros')
    mpc_dir = get_package_share_directory('mpc_cbf')

    # ── Launch arguments ──
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value='four_wall.yaml',
        description='Map YAML file name (in meco_truck_sim/maps/)')

    # ── Paths ──
    map_file_path = PathJoinSubstitution([
        sim_dir, 'maps', LaunchConfiguration('map_file')])

    nav2_params = os.path.join(
        bringup_dir, 'config', 'nav2_planner_params_lidar.yaml')
    mpc_params = os.path.join(mpc_dir, 'config', 'mpc_cbf_params_scan.yaml')
    firi_params = os.path.join(firi_dir, 'config', 'firi_scan_params.yaml')


    # ================================================================
    # 1. MAP NODE
    # ================================================================
    map_node = Node(
        package='meco_truck_sim',
        executable='live_gridmap_node',
        name='live_gridmap_node',
        output='screen',
        parameters=[{'map_yaml_path': map_file_path}]
    )

    # ================================================================
    # 2. PCL FILTER NODE
    #    Subscribes /ouster/points → publishes /filtered_cloud,
    #    /filtered_scan
    # ================================================================
    pcl_filter_node = Node(
        package='firi_ros',
        executable='pcl_filter_node',
        name='pcl_filter_node',
        output='screen',
        # parameters=[{
        #     'cloud_topic': '/ouster/points',
        #     'scan_frame': 'base_link',
        #     'voxel_size': 0.05,
        #     'height_min': 0.3,
        #     'height_max': 1.5,
        #     'range_min': 0.15,
        #     'range_max': 4.0,
        #     'num_beams': 720,
        #     'robot_length': 0.65,
        #     'robot_width': 0.35,
        #     'footprint_offset_x': 0.265,
        #     'footprint_margin': 0.05,
        #     'outlier_radius': 0.2,
        #     'outlier_min_neighbors': 4,
        # }]
        parameters=[firi_params]
        )
    
    firi_node = Node(
        package='firi_ros',
        # executable='firi_scan_node',
        # name='firi_scan_node',
        executable='firi_scan_experiment_node',
        name='firi_scan_experiment_node',
        output='screen',
        parameters=[firi_params]
    )

    ouster_tf_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tracker_to_robot',
            # arguments: x y z yaw pitch roll frame_id child_frame_id
            arguments=['0.155', '0', '0.25', '0', '0', '0', 'base_link', 'os_sensor']
        )

    # ================================================================
    # 3. NAV2 PLANNER SERVER (NavFn A*)
    # ================================================================
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params])

    planner_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_planner',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            {'autostart': True},
            {'node_names': ['planner_server']}])

    # ================================================================
    # 4. GOAL-TO-PLAN BRIDGE
    # ================================================================
    bridge_node = Node(
        package='mpc_cbf',
        executable='goal_to_plan_bridge',
        name='goal_to_plan_bridge',
        output='screen')

    # ================================================================
    # 5. MPC-CBF NODE
    # ================================================================
    mpc_node = Node(
        package='mpc_cbf',
        executable='mpc_cbf_node',
        name='mpc_cbf_node',
        output='screen',
        parameters=[mpc_params])

    # ================================================================
    # LAUNCH
    # ================================================================
    return LaunchDescription([
        map_file_arg,

        # Map
        map_node,

        # Control
        mpc_node,
        
        # Perception
        pcl_filter_node,
        firi_node,
        ouster_tf_node,

        # Planning
        planner_server,
        planner_lifecycle,
        bridge_node,
    ])