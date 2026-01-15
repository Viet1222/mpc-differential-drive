import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
import numpy as np
import math
import csv
import os

# QUAN TRỌNG: Dùng Solver NÂNG CAO (Có vật cản)
from mpc_bot.mpc_solver_obstacle import MPCObstacleSolver

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class ScenarioFigure8Obstacle(Node):
    def __init__(self):
        super().__init__('scenario_figure8_obstacle')
        
        # Cấu hình MPC
        self.dt = 0.1
        self.N = 40  # Cần N lớn (40) để nhìn thấy vật cản từ xa và lách mượt
        self.solver = MPCObstacleSolver(self.N, self.dt)
        
        self.get_logger().info(" Kịch bản: HÌNH SỐ 8 + VẬT CẢN (Static) đã khởi động!")
        
        # --- CẤU HÌNH VẬT CẢN ---
        # Đặt vật cản nằm chắn ngay trên đường đi của nhánh số 8
        # Hình 8 có biên độ A=2.0. Ta đặt vật tại (1.0, 0.4) là điểm xe chắc chắn đi qua.
        self.obs_x = 1.0
        self.obs_y = 0.86
        self.obs_r = 0.4 # Bán kính vật cản (To một chút để dễ nhìn)
        self.obs_param = np.array([self.obs_x, self.obs_y, self.obs_r + 0.35]) # +0.35 là an toàn
        
        # --- GHI LOG ---
        self.csv_file_path = os.path.expanduser('~/mpc_data_figure8_obstacle.csv')
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        header = ['time', 'x_ref', 'y_ref', 'theta_ref', 'x_act', 'y_act', 'theta_act', 
                  'v_cmd', 'omega_cmd', 'v_left', 'v_right', 'cost_val']
        self.csv_writer.writerow(header)

        # ROS Comm
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_ref_path = self.create_publisher(Path, '/mpc/reference_path', 10)
        self.pub_pred_path = self.create_publisher(Path, '/mpc/predicted_path', 10)
        self.pub_marker = self.create_publisher(Marker, '/mpc/obstacle_marker', 10)
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.current_state = np.array([0.0, 0.0, 0.0]) 
        self.got_odom = False
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.initial_time = self.start_time
        
        # Thông số xe
        self.L_wheel = 0.35

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_state = np.array([x, y, theta])
        self.got_odom = True

    def generate_figure_8(self):
        # Tạo quỹ đạo hình số 8 chuẩn
        t_now = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        ref_matrix = np.zeros((3, self.N))
        A = 2.0; omega = 0.25 
        for i in range(self.N):
            t = t_now + i * self.dt
            x_ref = A * np.sin(omega * t)
            y_ref = A * np.sin(omega * t) * np.cos(omega * t)
            dx = A * omega * np.cos(omega * t)
            dy = A * omega * (np.cos(omega * t)**2 - np.sin(omega * t)**2)
            theta_ref = np.arctan2(dy, dx)
            
            ref_matrix[0, i] = x_ref
            ref_matrix[1, i] = y_ref
            ref_matrix[2, i] = theta_ref
        return ref_matrix

    def publish_obstacle_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = 0
        marker.pose.position.x = self.obs_x
        marker.pose.position.y = self.obs_y
        marker.pose.position.z = 0.0
        marker.scale.x = (self.obs_r + 0.35) * 2 # Vẽ vùng an toàn (Safety bubble)
        marker.scale.y = (self.obs_r + 0.35) * 2
        marker.scale.z = 0.1
        marker.color.a = 0.5; marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
        self.pub_marker.publish(marker)
        
        # Vẽ thêm lõi vật cản (nhỏ hơn, đậm hơn)
        core = Marker()
        core.header = marker.header
        core.type = Marker.CYLINDER
        core.id = 1
        core.pose.position.x = self.obs_x
        core.pose.position.y = self.obs_y
        core.pose.position.z = 0.25
        core.scale.x = self.obs_r * 2
        core.scale.y = self.obs_r * 2
        core.scale.z = 0.5
        core.color.a = 1.0; core.color.r = 0.5; core.color.g = 0.0; core.color.b = 0.0
        self.pub_marker.publish(core)

    def visualize_path(self, trajectory_matrix, publisher):
        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(trajectory_matrix.shape[1]):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(trajectory_matrix[0, i])
            pose.pose.position.y = float(trajectory_matrix[1, i])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0 
            path_msg.poses.append(pose)
        publisher.publish(path_msg)

    def control_loop(self):
        if not self.got_odom: return

        # 1. Lấy quỹ đạo mẫu (Số 8)
        ref_traj = self.generate_figure_8()
        
        # Xử lý góc xoay vòng
        full_theta = np.concatenate(([self.current_state[2]], ref_traj[2, :]))
        full_theta_unwrapped = np.unwrap(full_theta)
        ref_traj[2, :] = full_theta_unwrapped[1:]
        
        # 2. GỌI SOLVER (Truyền thêm tham số vật cản)
        # Solver sẽ tự cân nhắc: Bám ref_traj HAY LÀ Né obs_param
        u_opt, x_pred = self.solver.solve(self.current_state, ref_traj, self.obs_param)
        
        # 3. Điều khiển
        v_cmd = float(u_opt[0])
        w_cmd = float(u_opt[1])
        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.pub_cmd.publish(cmd)
        
        # 4. Hiển thị
        self.visualize_path(ref_traj, self.pub_ref_path) # Đường xanh (Mơ ước)
        self.visualize_path(x_pred, self.pub_pred_path)  # Đường đỏ (Thực tế sắp đi)
        self.publish_obstacle_marker()
        
        # 5. Ghi Log (Tính luôn v_left, v_right cho tiện)
        current_time = self.get_clock().now().nanoseconds / 1e9 - self.initial_time
        v_L = v_cmd - (w_cmd * self.L_wheel / 2)
        v_R = v_cmd + (w_cmd * self.L_wheel / 2)
        
        # Lưu ý: Cần lấy cost từ solver (nếu bạn đã sửa solver theo hướng dẫn trước). 
        # Nếu chưa sửa solver để trả về cost, tạm thời để cost = 0.
        cost_val = 0.0 
        
        self.csv_writer.writerow([
            f"{current_time:.2f}",
            f"{ref_traj[0,0]:.3f}", f"{ref_traj[1,0]:.3f}", f"{ref_traj[2,0]:.3f}",
            f"{self.current_state[0]:.3f}", f"{self.current_state[1]:.3f}", f"{self.current_state[2]:.3f}",
            f"{v_cmd:.3f}", f"{w_cmd:.3f}",
            f"{v_L:.3f}", f"{v_R:.3f}",
            f"{cost_val:.4f}"
        ])

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info("📁 Đã lưu file mpc_data_figure8_obstacle.csv!")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ScenarioFigure8Obstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
