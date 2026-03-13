import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # Locate the installed config file
    # pkg_dir = get_package_share_directory('firi_ros')
    # params_file = os.path.join(pkg_dir, 'config', 'firi_params.yaml')
    # params_file = os.path.join(pkg_dir, 'config', 'firi_path_params.yaml')

    firi_node = Node(
        package='firi_ros',
        executable='firi_scan_node',
        name='firi_scan_node',
        output='screen',
        # parameters=[params_file]
    )

    pcl_node = Node(
        package='firi_ros',
        executable='pcl_filter_node',
        name='pcl_filter_node',
        output='screen',
        # parameters=[params_file]
    )
    

    tf_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tracker_to_robot',
            # arguments: x y z yaw pitch roll frame_id child_frame_id
            arguments=['0.155', '0', '0.48', '0', '0', '0', 'base_link', 'os_sensor']
        )
    
    return LaunchDescription([
        # firi_node,
        pcl_node,
        tf_node
    ])