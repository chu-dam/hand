from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    grasp_pkg = FindPackageShare("dg5f_grasp_control")
    driver_pkg = FindPackageShare("dg5f_s_driver")

    common_param_file = PathJoinSubstitution([
        grasp_pkg, "config", "grasp_real_common.yaml",
    ])
    default_param_file = PathJoinSubstitution([
        grasp_pkg, "config", "grasp_real_left_gains.yaml",
    ])
    effort_launch = PathJoinSubstitution([
        driver_pkg,
        "launch",
        "dg5f_s_left_effort_controller.launch.py",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("param_file", default_value=default_param_file),
        IncludeLaunchDescription(PythonLaunchDescriptionSource([effort_launch])),
        Node(
            package="dg5f_grasp_control",
            executable="grasp_real",
            name="grasp_real",
            output="screen",
            parameters=[common_param_file, LaunchConfiguration("param_file")],
        ),
    ])
