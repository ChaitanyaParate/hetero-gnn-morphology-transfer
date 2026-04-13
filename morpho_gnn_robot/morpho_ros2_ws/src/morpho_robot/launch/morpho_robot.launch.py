"""
morpho_robot.launch.py
Full system launch for Morphology-Generalizable Robotic Control
ROS2 Jazzy + Gazebo Harmonic

Directory layout assumed:
  your_ws/
    src/
      morpho_robot/
        launch/
          morpho_robot.launch.py   <-- this file
        urdf/
          anymal.urdf
        worlds/
          robot_world.sdf
        config/
          bridge.yaml
        morpho_robot/
          vision_node.py
          llm_planner_node.py
          skill_translator_node.py
          gnn_policy_node.py
                    MLP_policy_node.py

Build and source before launching:
  colcon build --symlink-install
  source install/setup.bash
  ros2 launch morpho_robot morpho_robot.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PKG = "morpho_robot"


def pkg_share(*rel_path: str) -> str:
    return os.path.join(get_package_share_directory(PKG), *rel_path)


# ---------------------------------------------------------------------------
# Launch Description
# ---------------------------------------------------------------------------


def generate_launch_description():

    # -----------------------------------------------------------------------
    # Launch arguments -- override any of these from the CLI
    # e.g.: ros2 launch morpho_robot morpho_robot.launch.py urdf:=hexapod.urdf
    # -----------------------------------------------------------------------

    urdf_arg = DeclareLaunchArgument(
        "urdf",
        default_value="anymal.urdf",
        description="URDF filename inside morpho_robot/urdf/",
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value="warehouse_world.sdf",
        description="SDF world filename inside morpho_robot/worlds/",
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="false",
        description="Launch RViz2 alongside Gazebo",
    )

    llm_model_arg = DeclareLaunchArgument(
        "llm_model",
        default_value="qwen2.5:7b",
        description="LLM model ID string passed to llm_planner_node",
    )

    mlp_checkpoint_arg = DeclareLaunchArgument(
        "mlp_checkpoint",
        default_value="/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_MLP/checkpoints/mlp_ppo_6303744.pt",
        description="Absolute path to .pt MLP checkpoint. Empty = random init.",
    )
#mlp_ppo_6303744.pt
    speed_multiplier_arg = DeclareLaunchArgument(
        "speed_multiplier",
        default_value="1.0",
        description="Global stride length multiplier (higher = faster)",
    )

    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="ROS2 logger level: debug | info | warn | error",
    )

    spawn_yaw_arg = DeclareLaunchArgument(
        "spawn_yaw",
        default_value="0.0",
        description="Robot spawn yaw in radians",
    )

    action_remap_arg = DeclareLaunchArgument(
        "action_remap",
        default_value="none",
        description="Rotate gait direction: none | rotate_cw | rotate_ccw",
    )

    odom_in_base_frame_arg = DeclareLaunchArgument(
        "odom_in_base_frame",
        default_value="1",
        description="Treat /odom twist as base frame (1) or world frame (0). "
                    "Set to 0 because Gazebo /odom twist is in world frame, and we need the node to rotate it to body frame for the policy.",
    )

    # -----------------------------------------------------------------------
    # Resolved paths
    # -----------------------------------------------------------------------

    urdf_path = PathJoinSubstitution([FindPackageShare(PKG), "urdf", LaunchConfiguration("urdf")])

    world_path = PathJoinSubstitution([FindPackageShare(PKG), "worlds", LaunchConfiguration("world")])

    # -----------------------------------------------------------------------
    # 1. Robot description (robot_state_publisher)
    #    Reads URDF, publishes /robot_description and TF
    # -----------------------------------------------------------------------

    robot_description_content = Command(["cat ", urdf_path])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description_content,
                "use_sim_time": True,
            }
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    # -----------------------------------------------------------------------
    # 2. Gazebo Harmonic (gz sim)
    #    ros_gz_sim provides the launch file wrapper.
    #    GZ_SIM_RESOURCE_PATH lets Gazebo find your URDF meshes.
    # -----------------------------------------------------------------------

    gz_resource_path = os.pathsep.join(
        [
            pkg_share(),                        # package root
            pkg_share("urdf"),                  # URDF meshes
            pkg_share("worlds"),                # SDF worlds
            pkg_share("config"),                # bridge.yaml
            pkg_share("meshes"),                # additional meshes
        ]
    )

    from launch.actions import SetEnvironmentVariable

    gazebo = [
        # ✅ Set environment variable globally
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=gz_resource_path,
        ),

        # ✅ Include Gazebo launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py",
                )
            ),
            launch_arguments={
                "gz_args": [world_path, " -r -v4"],
            }.items(),
        ),
    ]

    # -----------------------------------------------------------------------
    # 3. Spawn the robot into Gazebo
    #    Uses /robot_description topic published by robot_state_publisher.
    #    Delay 3 s to give Gazebo time to fully load the world.
    # -----------------------------------------------------------------------

    spawn_robot = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                name="spawn_robot",
                output="screen",
                arguments=[
                    "-topic", "/robot_description",
                    "-name",  "robot",
                    "-world", "robot_world",
                    "-x",     "0.0",
                    "-y",     "0.0",
                    "-z",     "0.5",   # Final stabilized height
                    "-Y",     LaunchConfiguration("spawn_yaw"),
                ],
            )
        ],
    )

    # -----------------------------------------------------------------------
    # 4. Gazebo <-> ROS2 topic bridge
    #    Maps gz topics to ROS2 topics.
    #    Config file: config/bridge.yaml
    #
    #    Minimal bridge.yaml content (create this file in your package):
    #    ----------------------------------------------------------------
    #    - ros_topic_name:  /joint_states
    #      gz_topic_name:   /world/robot_world/model/robot/joint_state
    #      ros_type_name:   sensor_msgs/msg/JointState
    #      gz_type_name:    gz.msgs.Model
    #      direction:       GZ_TO_ROS
    #
    #    - ros_topic_name:  /cmd_joint_torques
    #      gz_topic_name:   /world/robot_world/model/robot/joint_cmd
    #      ros_type_name:   std_msgs/msg/Float64MultiArray
    #      gz_type_name:    gz.msgs.Double_V
    #      direction:       ROS_TO_GZ
    #
    #    - ros_topic_name:  /camera/image_raw
    #      gz_topic_name:   /world/robot_world/model/robot/link/camera_link/sensor/camera/image
    #      ros_type_name:   sensor_msgs/msg/Image
    #      gz_type_name:    gz.msgs.Image
    #      direction:       GZ_TO_ROS
    #
    #    - ros_topic_name:  /odom
    #      gz_topic_name:   /model/robot/odometry
    #      ros_type_name:   nav_msgs/msg/Odometry
    #      gz_type_name:    gz.msgs.Odometry
    #      direction:       GZ_TO_ROS
    #    ----------------------------------------------------------------

    gz_bridge = TimerAction(
        period=4.0,   # wait for Gazebo + spawn to finish
        actions=[
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="gz_ros_bridge",
                output="screen",
                parameters=[
                    {
                        "config_file": pkg_share("config", "bridge.yaml"),
                        "use_sim_time": True,
                    }
                ],
                arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
            )
        ],
    )

    # -----------------------------------------------------------------------
    # 5. Vision node
    #    Subscribes:  /camera/image_raw  (sensor_msgs/Image)
    #    Publishes:   /scene_graph        (std_msgs/String, JSON)
    #
    #    Runs YOLOv8n inference. Outputs:
    #    {"objects": [{"label": "box", "position": [x,y,z], "distance": 1.2}]}
    # -----------------------------------------------------------------------

    vision_node = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/.venv/bin/python',
                    '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/morpho_robot/vision_node.py',
                    '--yolo_model', 'yolov8s.pt',
                    '--conf', '0.4'
                ],
                output='screen',
            )
        ],
    )

    # -----------------------------------------------------------------------
    # 6. LLM planner node
    #    Subscribes:  /scene_graph        (std_msgs/String, JSON)
    #    Publishes:   /llm_action         (std_msgs/String, JSON)
    #
    #    Calls local Ollama model every 5 s (configurable).
    #    Output schema:
    #    {"skill": "navigate_to", "target": "box_1", "params": {}}
    # -----------------------------------------------------------------------

    llm_planner_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package=PKG,
                executable="llm_planner_node",
                name="llm_planner_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time":       True,
                        "llm_model":          LaunchConfiguration("llm_model"),
                        "replan_interval_s":  5.0,
                        "scene_graph_topic":  "/scene_graph",
                        "action_topic":       "/llm_action",
                        "speed_multiplier":   LaunchConfiguration("speed_multiplier"),
                        "max_tokens":         256,
                        "task":               "navigate to the goal",
                    }
                ],
                arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
            )
        ],
    )

    # -----------------------------------------------------------------------
    # 7. Skill translator node
    #    Subscribes:  /llm_action         (std_msgs/String, JSON)
    #                 /scene_graph        (std_msgs/String, JSON)
    #    Publishes:   /goal_pose          (geometry_msgs/PoseStamped)
    #                 /active_skill       (std_msgs/String)
    #
    #    Maps {"skill":"navigate_to","target":"box_1"} to a goal position
    #    extracted from the scene graph, then publishes it for GNN policy.
    # -----------------------------------------------------------------------

    skill_translator_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package=PKG,
                executable="skill_translator_node",
                name="skill_translator_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time":        True,
                        "llm_action_topic":    "/llm_action",
                        "scene_graph_topic":   "/scene_graph",
                        "goal_pose_topic":     "/goal_pose",
                        "active_skill_topic":  "/active_skill",
                    }
                ],
                arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
            )
        ],
    )

    # -----------------------------------------------------------------------
    # 8. MLP policy node
    #    Subscribes:  /joint_states       (sensor_msgs/JointState)
    #                 /odom               (nav_msgs/Odometry)
    #    Publishes:   /cmd_joint_torques  (std_msgs/Float64MultiArray)
    #
    #    Runs at 20 Hz. Loads PyTorch MLP checkpoint.
    #    If mlp_checkpoint is empty, runs with random weights (sanity check).
    # -----------------------------------------------------------------------

    mlp_policy_node = TimerAction(
        period=7.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/.venv/bin/python',
                    '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/morpho_robot/MLP_policy_node.py',
                    '--checkpoint',
                    LaunchConfiguration("mlp_checkpoint"),
                    '--urdf',
                    PathJoinSubstitution([FindPackageShare(PKG), "urdf", LaunchConfiguration("urdf")]),
                    '--device', 'cuda',
                    '--action_remap', LaunchConfiguration("action_remap"),
                    '--odom_in_base_frame', LaunchConfiguration("odom_in_base_frame"),
                    '--speed_multiplier', LaunchConfiguration("speed_multiplier"),
                ],
                output='screen',
            )
        ],
    )

    #    gnn_policy_node = TimerAction(
    #     period=2.5,
    #     actions=[
    #         ExecuteProcess(
    #             cmd=[
    #                 '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/.venv/bin/python',
    #                 '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/morpho_robot/gnn_policy_node.py',
    #                 '--checkpoint',
    #                 '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location/checkpoints/gnn_ppo_4272128.pt',
    #                 '--urdf',
    #                 '/mnt/newvolume/Programming/Python/Deep_Learning/'
    #                 'Relational_Bias_for_Morphological_Generalization/'
    #                 'morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/urdf/anymal.urdf',
    #                 '--device', 'cuda',
    #             ],
    #             output='screen',
    #         )
    #     ],
    # )

    # -----------------------------------------------------------------------
    # 9. Optional RViz2
    # -----------------------------------------------------------------------

    rviz_config = pkg_share("config", "morpho_robot.rviz")

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------

    return LaunchDescription(
        [
            # Args
            urdf_arg,
            world_arg,
            use_rviz_arg,
            llm_model_arg,
            mlp_checkpoint_arg,
            action_remap_arg,
            odom_in_base_frame_arg,
            speed_multiplier_arg,
            log_level_arg,
            spawn_yaw_arg,
            # Nodes -- order matters due to TimerAction delays
            robot_state_publisher,  # immediate
            *gazebo,                 # immediate
            spawn_robot,            # +2 s
            gz_bridge,              # +4 s
            vision_node,            # +5 s
            llm_planner_node,       # +5 s
            skill_translator_node,  # +5 s
            mlp_policy_node,        # +2.5 s
            rviz,                   # conditional
        ]
    )
