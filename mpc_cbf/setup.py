from setuptools import setup
import os
from glob import glob

package_name = 'mpc_cbf'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # Register package with ament index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Package manifest
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alexglzg',
    maintainer_email='alex_gg97@hotmail.com',
    description='MPC-CBF controller for bicycle-model AMR',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mpc_cbf_node = mpc_cbf.mpc_cbf_node:main',
            'goal_to_plan_bridge = mpc_cbf.goal_to_plan_bridge:main',
        ],
    },
)