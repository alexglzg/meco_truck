#!/usr/bin/env python3
"""
truck simulation launch file
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Get package directory
    pkg_dir = get_package_share_directory('meco_truck_sim')
    
    # Path to URDF file
    urdf_file = os.path.join(pkg_dir, 'urdf', 'truck.urdf')
    
    # Fallback: try relative path if package path doesn't work
    if not os.path.exists(urdf_file):
        urdf_file = os.path.join(
            os.path.dirname(__file__), '..', 'urdf', 'truck.urdf'
        )
    
    # Read URDF content
    with open(urdf_file, 'r') as file:
        robot_description_content = file.read()
    
    # Robot State Publisher - publishes the robot model to RViz
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': False,
            'publish_frequency': 30.0

        }],
        output='screen'
    )

    # Joint State Publisher - publishes default joint positions
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen'
    )
    
    # truck Simulator
    simulator = Node(
        package='meco_truck_sim',
        executable='simulation_node',
        name='meco_truck_simulator',
        output='screen'
    )
    
    # Static transform: map -> odom (if needed)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_broadcaster',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    # joint_state_publisher = Node(
    #     package='joint_state_publisher_gui',
    #     executable='joint_state_publisher_gui',
    #     name='joint_state_publisher_gui',
    #     output='screen',
    # )

        
    return LaunchDescription([
        robot_state_publisher,
        # joint_state_publisher,
        simulator,
        static_tf,  # Uncomment if you need odom frame
    ])