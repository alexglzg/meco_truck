import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. Get the package directory
    pkg_dir = get_package_share_directory('meco_truck_sim') 
        
    # 2. Declare the argument (defaulting to the filename only)
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        # default_value='intersection_roundabout.yaml',
        # default_value='intersection.yaml',
        # default_value='t_junction.yaml',
        default_value='parking_two.yaml',
        description='Name of the map file to load'
    )

    # 3. Join the paths safely
    # This creates: /share/meco_truck_sim/maps/t_junction.yaml
    map_file_path = PathJoinSubstitution([
        pkg_dir,
        'maps',
        LaunchConfiguration('map_file')
    ])

    # 4. Map Server Node
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_file_path}]
    )

    # 5. Lifecycle Manager
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': ['map_server']}
        ]
    )

    return LaunchDescription([
        map_file_arg,
        map_server_node,
        lifecycle_manager_node
    ])