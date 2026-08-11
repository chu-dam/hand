from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("dg5f_grasp_control")
    common_param_file = PathJoinSubstitution([
        pkg_share, "config", "grasp_real_common.yaml",
    ])
    default_param_file = PathJoinSubstitution([
        pkg_share, "config", "grasp_real_left_gains.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("param_file", default_value=default_param_file),
        Node(
            package="dg5f_grasp_control",
            executable="grasp_real",
            name="grasp_real",
            output="screen",
            parameters=[common_param_file, LaunchConfiguration("param_file")],
        ),
    ])
