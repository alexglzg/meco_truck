"""
Full system bringup for the truck AMR with MPC-CBF + FIRI + Nav2 planner.

Launches:
  1. Map server + lifecycle manager
  2. Truck simulator
  3. Robot state publisher (URDF → TF)
  4. FIRI node (polytope generation)
  5. MPC-CBF node (controller)
  6. Nav2 planner server (NavFn A*) + lifecycle manager
  7. Goal-to-plan bridge (RViz2 goal → NavFn action → /plan)
  8. RViz2

Usage:
  ros2 launch truck_bringup bringup.launch.py
  ros2 launch truck_bringup bringup.launch.py map_file:=intersection_roundabout.yaml
  ros2 launch truck_bringup bringup.launch.py rviz:=false
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
        default_value='t_junction.yaml',
        description='Map YAML file name (in meco_truck_sim/maps/)')

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz2')

    # ── Paths ──
    map_file_path = PathJoinSubstitution([
        sim_dir, 'maps', LaunchConfiguration('map_file')])

    nav2_params = os.path.join(
        bringup_dir, 'config', 'nav2_planner_params.yaml')
    firi_params = os.path.join(firi_dir, 'config', 'firi_params.yaml')
    mpc_params = os.path.join(mpc_dir, 'config', 'mpc_cbf_params.yaml')
    rviz_config = os.path.join(bringup_dir, 'rviz', 'truck_bringup.rviz')
    urdf_file = os.path.join(sim_dir, 'urdf', 'truck.urdf')

    # Read URDF for robot_state_publisher
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # ================================================================
    # 1. MAP SERVER
    # ================================================================
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_file_path}])

    map_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            {'autostart': True},
            {'node_names': ['map_server']}])

    # ================================================================
    # 2. TRUCK SIMULATOR
    # ================================================================
    truck_sim = Node(
        package='meco_truck_sim',
        executable='simulation_node',
        name='truck_simulator',
        output='screen')

    # ================================================================
    # 3. ROBOT STATE PUBLISHER (URDF → TF)
    # ================================================================
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}])

    # ================================================================
    # 4. FIRI NODE
    # ================================================================
    firi_node = Node(
        package='firi_ros',
        executable='firi_grid_node',
        name='firi_grid_node',
        output='screen',
        parameters=[firi_params])

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
    # 6. NAV2 PLANNER SERVER (NavFn A*)
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
    # 7. GOAL-TO-PLAN BRIDGE
    #    Subscribes /goal_pose → calls NavFn action → publishes /plan
    # ================================================================
    bridge_node = Node(
        package='mpc_cbf',
        executable='goal_to_plan_bridge',
        name='goal_to_plan_bridge',
        output='screen')

    # ================================================================
    # 8. RVIZ2
    # ================================================================
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('rviz')))

    # ================================================================
    # LAUNCH
    # ================================================================
    return LaunchDescription([
        # map_file_arg,
        # rviz_arg,

        # Infrastructure
        # map_server,
        # map_lifecycle,
        # robot_state_pub,

        # Simulator
        # truck_sim,

        # Perception + Control
        # firi_node,
        # mpc_node,

        # Planning
        planner_server,
        planner_lifecycle,
        bridge_node,

        # Visualization
        # rviz,
    ])