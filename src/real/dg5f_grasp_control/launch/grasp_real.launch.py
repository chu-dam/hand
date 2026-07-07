from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("dg5f_grasp_control")
    default_param_file = PathJoinSubstitution([pkg_share, "config", "grasp_real.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("param_file", default_value=default_param_file),
        Node(
            package="dg5f_grasp_control",
            executable="grasp_real",
            name="grasp_real",
            output="screen",
            parameters=[LaunchConfiguration("param_file")],
        ),
    ])
