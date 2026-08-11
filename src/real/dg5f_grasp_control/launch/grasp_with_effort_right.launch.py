from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    right_launch = PathJoinSubstitution([
        FindPackageShare("dg5f_grasp_control"),
        "launch",
        "grasp_right_compensation.launch.py",
    ])

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([right_launch]),
            launch_arguments={
                "fric_scale": "1.0",
                "hand_limit": "7.5",
                "start_teaching_mode": "false",
            }.items(),
        ),
    ])
