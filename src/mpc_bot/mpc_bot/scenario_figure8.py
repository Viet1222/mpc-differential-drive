import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import math
import csv # Thêm thư viện CSV
import os  # Thêm thư viện OS

# Dùng Solver Cơ bản
from mpc_bot.mpc_solver import MPCSolver

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class ScenarioFigure8(Node):
    def __init__(self):
        super().__init__('scenario_figure8')
        
        self.dt = 0.1
        self.N = 20
        self.solver = MPCSolver(self.N, self.dt)
        self.get_logger().info("  Kịch bản Hình Số 8  đã chạy!")
        
        # --- CẤU HÌNH GHI FILE CSV (Tích hợp luôn vào đây) ---
        self.csv_file_path = os.path.expanduser('~/mpc_data_figure8.csv')
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Ghi Header
        header = ['time', 'x_ref', 'y_ref', 'theta_ref', 'x_act', 'y_act', 'theta_act', 'v_cmd', 'omega_cmd']
        self.csv_writer.writerow(header)
        self.get_logger().info(f"💾 Dữ liệu sẽ được ghi vào: {self.csv_file_path}")

        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_ref_path = self.create_publisher(Path, '/mpc/reference_path', 10)
        self.pub_pred_path = self.create_publisher(Path, '/mpc/predicted_path', 10)
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.current_state = np.array([0.0, 0.0, 0.0]) 
        self.got_odom = False
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.initial_time = self.start_time

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_state = np.array([x, y, theta])
        self.got_odom = True

    def generate_figure_8(self):
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

        ref_traj = self.generate_figure_8()
        
        full_theta = np.concatenate(([self.current_state[2]], ref_traj[2, :]))
        full_theta_unwrapped = np.unwrap(full_theta)
        ref_traj[2, :] = full_theta_unwrapped[1:]
        
        u_opt, x_pred = self.solver.solve(self.current_state, ref_traj)
        
        v_cmd = float(u_opt[0])
        w_cmd = float(u_opt[1])
        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.pub_cmd.publish(cmd)
        
        self.visualize_path(ref_traj, self.pub_ref_path)
        self.visualize_path(x_pred, self.pub_pred_path)
        
        # --- GHI LOG ---
        current_time = self.get_clock().now().nanoseconds / 1e9 - self.initial_time
        self.csv_writer.writerow([
            f"{current_time:.2f}",
            f"{ref_traj[0,0]:.3f}", f"{ref_traj[1,0]:.3f}", f"{ref_traj[2,0]:.3f}",
            f"{self.current_state[0]:.3f}", f"{self.current_state[1]:.3f}", f"{self.current_state[2]:.3f}",
            f"{v_cmd:.3f}", f"{w_cmd:.3f}"
        ])

    def destroy_node(self):
        self.csv_file.close() # Đóng file khi tắt node
        self.get_logger().info("📁 Đã lưu file CSV thành công!")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ScenarioFigure8()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
