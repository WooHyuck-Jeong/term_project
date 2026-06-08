from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'neural_rrt_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name,
            ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),

        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='woohyuck',
    maintainer_email='woohyuck@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'rrt_star = neural_rrt_planner.planners.test_rrt_star:main',
            'rrt_star_node = neural_rrt_planner.rrt_star_node:main',
            'informed_rrt_star_test = neural_rrt_planner.planners.test_informed_rrt_star:main',
            'informed_rrt_star_node = neural_rrt_planner.informed_rrt_star_node:main',
            'generate_sampling_dataset = neural_rrt_planner.learning.generate_sampling_dataset:main',
            'neural_rrt_star_node = neural_rrt_planner.neural_rrt_star_node:main',
            'neural_rrt_star_node_v2 = neural_rrt_planner.neural_rrt_star_node_v2:main',
        ],
    },
)