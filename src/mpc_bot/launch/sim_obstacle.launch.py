import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_mpc_bot = get_package_share_directory('mpc_bot')

    # 1. Gọi file sim nền (Mở Gazebo + Spawn Robot)
    base_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_mpc_bot, 'launch', 'sim.launch.py')
        )
    )

    # 2. Định nghĩa đường dẫn đến file vật cản
  
    obstacle_file = os.path.join(pkg_mpc_bot, 'models', 'obstacle.sdf')

    # 3. Spawn Vật cản (Cục đá) tại vị trí CHẮN ĐƯỜNG Hình số 8
    spawn_obstacle = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', obstacle_file,
            '-name', 'static_obstacle',
            
            # --- CẬP NHẬT TỌA ĐỘ MỚI ---
            '-x', '1.0',  # Đặt tại x=1.0 (Nằm trong biên độ +-2.0 của hình số 8)
            '-y', '0.86',  # Đặt tại y=0.4 (Chắn ngang đường cong)
            '-z', '0.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        base_sim,
        spawn_obstacle
    ])
