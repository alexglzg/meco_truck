import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # Locate the installed config file
    pkg_dir = get_package_share_directory('firi_ros')
    params_file = os.path.join(pkg_dir, 'config', 'firi_params.yaml')
    # params_file = os.path.join(pkg_dir, 'config', 'firi_path_params.yaml')

    firi_node = Node(
        package='firi_ros',
        executable='firi_grid_node',
        name='firi_grid_node',
        output='screen',
        parameters=[params_file]
    )
    
    firi_path_node = Node(
        package='firi_ros',
        executable='firi_path_grid_node',
        name='firi_path_grid_node',
        output='screen',
        parameters=[params_file]
    )
    return LaunchDescription([
        firi_node,
        # firi_path_node
    ])