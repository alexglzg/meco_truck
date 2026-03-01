import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    pkg_dir = get_package_share_directory('mpc_cbf')
    params_file = os.path.join(pkg_dir, 'config', 'mpc_cbf_params.yaml')

    mpc_node = Node(
        package='mpc_cbf',
        executable='mpc_cbf_node',
        name='mpc_cbf_node',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        mpc_node
    ])
