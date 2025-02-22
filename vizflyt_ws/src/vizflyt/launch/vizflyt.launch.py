from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():

    # Drone Pose Publisher Node
    drone_pose_pub_node = Node(
        package='vizflyt',
        executable='drone_pose_pub',
        name='drone_pose_pub',
        output='screen'
    )
    
    # Render Node
    render_node = Node(
        package='vizflyt',
        executable='render_node',
        name='render_node',
        output='screen'
    )

    return LaunchDescription([
        
        drone_pose_pub_node,
        render_node, 
    ])
