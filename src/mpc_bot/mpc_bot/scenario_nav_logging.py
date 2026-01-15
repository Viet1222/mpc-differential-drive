import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import math
import csv # Thư viện ghi file
import os

# Dùng Solver Cơ bản
from mpc_bot.mpc_solver import MPCSolver

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class ScenarioNavLogging(Node):
    def __init__(self):
        super().__init__('scenario_nav_logging')
        
        self.dt = 0.1
        self.N = 20
        self.solver = MPCSolver(self.N, self.dt)
        self.get_logger().info("💾 Kịch bản GHI DỮ LIỆU (Logging) đã sẵn sàng!")
        
        # --- CẤU HÌNH GHI FILE CSV ---
        # File sẽ được lưu tại thư mục Home của user (~/mpc_data.csv)
        self.csv_file_path = os.path.expanduser('~/mpc_data.csv')
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Ghi dòng tiêu đề (Header)
        header = ['time', 'x_ref', 'y_ref', 'theta_ref', 'x_act', 'y_act', 'theta_act', 'v_cmd', 'omega_cmd']
        self.csv_writer.writerow(header)
        self.get_logger().info(f"📁 Đang ghi dữ liệu vào: {self.csv_file_path}")
        # -----------------------------

        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_ref_path = self.create_publisher(Path, '/mpc/reference_path', 10)
        self.pub_pred_path = self.create_publisher(Path, '/mpc/predicted_path', 10)
        
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.current_state = np.array([0.0, 0.0, 0.0]) 
        self.got_odom = False
        self.has_goal = False
        self.goal_state = np.array([0.0, 0.0, 0.0])
        self.is_reached = False
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.initial_time = self.start_time # Mốc thời gian bắt đầu chạy

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_state = np.array([x, y, theta])
        self.got_odom = True

    def goal_callback(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        q = msg.pose.orientation
        gtheta = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.goal_state = np.array([gx, gy, gtheta])
        self.has_goal = True
        self.is_reached = False
        self.get_logger().info(f"🎯 Nhận mục tiêu: X={gx:.2f}, Y={gy:.2f}")

    def generate_path_to_goal(self):
        # (Logic tạo đường dẫn giống hệt scenario_nav cũ)
        ref_matrix = np.zeros((3, self.N))
        dx = self.goal_state[0] - self.current_state[0]
        dy = self.goal_state[1] - self.current_state[1]
        dist_to_goal = math.sqrt(dx**2 + dy**2)
        
        if dist_to_goal < 0.1:
            self.is_reached = True
            for i in range(self.N): ref_matrix[:, i] = self.goal_state
            return ref_matrix

        v_des = 0.5
        angle_to_goal = math.atan2(dy, dx)
        for i in range(self.N):
            dist_future = v_des * (i * self.dt)
            if dist_future > dist_to_goal:
                ref_matrix[:, i] = self.goal_state
            else:
                ratio = dist_future / dist_to_goal
                ref_matrix[0, i] = self.current_state[0] + dx * ratio
                ref_matrix[1, i] = self.current_state[1] + dy * ratio
                if dist_to_goal > 0.5: ref_matrix[2, i] = angle_to_goal
                else: ref_matrix[2, i] = self.goal_state[2]
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
        
        # Nếu chưa có goal, vẫn ghi log (để thấy xe đứng yên) nhưng lệnh v=0
        if not self.has_goal:
            self.pub_cmd.publish(Twist())
            return

        ref_traj = self.generate_path_to_goal()
        
        full_theta = np.concatenate(([self.current_state[2]], ref_traj[2, :]))
        full_theta_unwrapped = np.unwrap(full_theta)
        ref_traj[2, :] = full_theta_unwrapped[1:]
        
        u_opt, x_pred = self.solver.solve(self.current_state, ref_traj)
        
        v_cmd = float(u_opt[0])
        w_cmd = float(u_opt[1])

        if self.is_reached:
            v_cmd, w_cmd = 0.0, 0.0
            
        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.pub_cmd.publish(cmd)
        
        self.visualize_path(ref_traj, self.pub_ref_path)
        self.visualize_path(x_pred, self.pub_pred_path)
        
        # --- GHI LOG ---
        # Chỉ ghi khi xe đang chạy hoặc vừa dừng
        current_time = self.get_clock().now().nanoseconds / 1e9 - self.initial_time
        
        # Lấy điểm tham chiếu đầu tiên (ngay tại thời điểm hiện tại)
        x_ref_now = ref_traj[0, 0]
        y_ref_now = ref_traj[1, 0]
        theta_ref_now = ref_traj[2, 0]
        
        self.csv_writer.writerow([
            f"{current_time:.2f}",
            f"{x_ref_now:.3f}", f"{y_ref_now:.3f}", f"{theta_ref_now:.3f}", # Tham chiếu
            f"{self.current_state[0]:.3f}", f"{self.current_state[1]:.3f}", f"{self.current_state[2]:.3f}", # Thực tế
            f"{v_cmd:.3f}", f"{w_cmd:.3f}" # Điều khiển
        ])

    # Đóng file khi tắt node
    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info("📁 Đã lưu file CSV thành công!")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ScenarioNavLogging()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
