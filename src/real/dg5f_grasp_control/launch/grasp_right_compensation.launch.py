from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    grasp_pkg = FindPackageShare("dg5f_grasp_control")
    driver_pkg = FindPackageShare("dg5f_s_driver")
    param_file = PathJoinSubstitution([
        grasp_pkg,
        "config",
        "grasp_real_common.yaml",
    ])
    right_gain_file = PathJoinSubstitution([
        grasp_pkg,
        "config",
        "grasp_real_right_gains.yaml",
    ])
    effort_launch = PathJoinSubstitution([
        driver_pkg,
        "launch",
        "dg5f_s_right_effort_controller.launch.py",
    ])

    fric_scale = LaunchConfiguration("fric_scale")
    hand_limit = LaunchConfiguration("hand_limit")
    start_teaching_mode = LaunchConfiguration("start_teaching_mode")

    return LaunchDescription([
        DeclareLaunchArgument("fric_scale", default_value="0.0"),
        DeclareLaunchArgument("hand_limit", default_value="0.3"),
        DeclareLaunchArgument("start_teaching_mode", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([effort_launch]),
            launch_arguments={"fingertip_sensor": "true"}.items(),
        ),
        Node(
            package="dg5f_grasp_control",
            executable="grasp_real",
            name="grasp_real",
            output="screen",
            parameters=[
                param_file,
                right_gain_file,
                {
                    "hand_side": "right",
                    "start_teaching_mode": ParameterValue(
                        start_teaching_mode,
                        value_type=bool,
                    ),
                    "joint_state_topic": "/dg5f_s_right/joint_states",
                    "effort_topic": "/dg5f_s_right/effort_controller/commands",
                    "command_topic": "/dg5f_grasp_control/right/grasp_type",
                    "pose_topic": "/dg5f_grasp_control/right/pose_type",
                    "alpha1_topic": "/dg5f_grasp_control/right/alpha1_cmd",
                    "relative_translation_topic": "/dg5f_grasp_control/right/relative_translation_cmd",
                    "relative_rotation_deg_topic": "/dg5f_grasp_control/right/relative_rotation_deg_cmd",
                    "continuous_rotation_topic": "/dg5f_grasp_control/right/continuous_rotation_cmd",
                    "rotation_matrix_topic": "/dg5f_grasp_control/right/rotation_matrix_cmd",
                    "teaching_mode_topic": "/dg5f_grasp_control/right/teaching_mode",
                    "debug_topic": "/dg5f_grasp_control/right/debug",
                    "fric_scale": ParameterValue(fric_scale, value_type=float),
                    "hand_limit": ParameterValue(hand_limit, value_type=float),
                },
            ],
        ),
    ])
