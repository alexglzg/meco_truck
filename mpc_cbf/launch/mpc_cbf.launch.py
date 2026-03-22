import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory('mpc_cbf')
    mode = LaunchConfiguration('mode').perform(context)

    params = [os.path.join(pkg_dir, 'config', 'mpc_cbf_params.yaml')]
    if mode == 'scan':
        params.append(os.path.join(pkg_dir, 'config', 'mpc_cbf_params_scan.yaml'))

    mpc_node = Node(
        package='mpc_cbf',
        executable='mpc_cbf_node',
        name='mpc_cbf_node',
        output='screen',
        parameters=params,
    )
    return [mpc_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='grid',
            description='Parameter mode: grid (robot footprint only) or scan (real-world with sensor margins)'),
        OpaqueFunction(function=launch_setup),
    ])
