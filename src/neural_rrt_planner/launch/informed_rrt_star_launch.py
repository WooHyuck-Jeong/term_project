import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    package_share_dir = get_package_share_directory('neural_rrt_planner')

    rviz_config_path = os.path.join(
        package_share_dir,
        'rviz',
        'informed_rrt_star.rviz'
    )

    # rrt_star_node = Node(
    #     package='neural_rrt_planner',
    #     executable='rrt_star_node',
    #     name='rrt_star_node',
    #     output='screen'
    # )
    
    informed_rrt_star_node = Node(
        package= 'neural_rrt_planner',
        executable= 'informed_rrt_star_node',
        name= 'informed_rrt_star_node',
        output= 'screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]
    )

    return LaunchDescription([
        # rrt_star_node,
        informed_rrt_star_node,
        rviz_node
    ])