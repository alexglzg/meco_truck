from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'meco_truck_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*.pgm')),

        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),

        (os.path.join('share', package_name, 'urdf'),
         glob('urdf/*.urdf')),
        
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),

        (os.path.join('share', package_name, 'cfg'), glob('cfg/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='alex_gg97@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulation_node = meco_truck_sim.simulation_node:main',
        ],
    },
)
