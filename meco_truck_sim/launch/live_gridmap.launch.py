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
        default_value='four_wall.yaml',
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
    map_node = Node(
        package='meco_truck_sim',
        executable='live_gridmap_node',
        name='live_gridmap_node',
        output='screen',
        parameters=[{'map_yaml_path': map_file_path}]
    )

    return LaunchDescription([
        map_file_arg,
        map_node
    ])


# ros2 launch meco_truck_sim live_gridmap.launch.py map_file:=intersection_roundabout.yaml
