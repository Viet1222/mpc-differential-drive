import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import math

# --- TÁI SỬ DỤNG BỘ NÃO CŨ ---
from mpc_bot.mpc_solver import MPCSolver

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class MPCScenarioSquare(Node):
    def __init__(self):
        super().__init__('scenario_square') # Tên node khác
        
        # 1. Khởi tạo MPC (Dùng lại y nguyên logic cũ)
        self.dt = 0.1
        self.N = 20
        self.solver = MPCSolver(self.N, self.dt)
        self.get_logger().info("✅ Kịch bản HÌNH VUÔNG đã sẵn sàng!")

        # 2. Setup ROS
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_ref_path = self.create_publisher(Path, '/mpc/reference_path', 10)
        self.pub_pred_path = self.create_publisher(Path, '/mpc/predicted_path', 10)
        
        self.timer = self.create_timer(self.dt, self.control_loop)

        # 3. Biến trạng thái
        self.current_state = np.array([0.0, 0.0, 0.0]) 
        self.got_odom = False
        self.start_time = self.get_clock().now().nanoseconds / 1e9

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_state = np.array([x, y, theta])
        self.got_odom = True

    # --- KHÁC BIỆT DUY NHẤT: Hàm tạo đường Hình Vuông ---
    def generate_square_path(self):
        t_now = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        ref_matrix = np.zeros((3, self.N))
        
        # Cấu hình hình vuông: Cạnh 2m
        # Chu kỳ: 40 giây (Mỗi cạnh đi mất 10s)
        cycle_duration = 40.0
        side_duration = 10.0
        
        for i in range(self.N):
            t = (t_now + i * self.dt) % cycle_duration # Lặp lại vô tận
            
            # Logic chia 4 cạnh dựa trên thời gian
            if 0 <= t < side_duration: 
                # Cạnh 1: Đi dọc trục X (0 -> 2)
                x_ref = 2.0 * (t / side_duration)
                y_ref = 0.0
                theta_ref = 0.0
                
            elif side_duration <= t < 2*side_duration: 
                # Cạnh 2: Đi dọc trục Y (0 -> 2) tại x=2
                local_t = t - side_duration
                x_ref = 2.0
                y_ref = 2.0 * (local_t / side_duration)
                theta_ref = np.pi/2 # 90 độ
                
            elif 2*side_duration <= t < 3*side_duration: 
                # Cạnh 3: Đi lùi trục X (2 -> 0) tại y=2
                local_t = t - 2*side_duration
                x_ref = 2.0 - 2.0 * (local_t / side_duration)
                y_ref = 2.0
                theta_ref = np.pi # 180 độ
                
            else: 
                # Cạnh 4: Đi lùi trục Y (2 -> 0) tại x=0
                local_t = t - 3*side_duration
                x_ref = 0.0
                y_ref = 2.0 - 2.0 * (local_t / side_duration)
                theta_ref = -np.pi/2 # -90 độ

            ref_matrix[0, i] = x_ref
            ref_matrix[1, i] = y_ref
            ref_matrix[2, i] = theta_ref
            
        return ref_matrix

    def visualize_path(self, trajectory_matrix, publisher):
        # (Giữ nguyên hàm vẽ cũ)
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
        if not self.got_odom:
            return

        # 1. Tạo Reference (HÌNH VUÔNG)
        ref_traj = self.generate_square_path()
        
        # --- QUAN TRỌNG: Unwrap góc cho Hình Vuông ---
        # Hình vuông có các góc cua gắt 90 độ, unwrap là bắt buộc để không xoay vòng
        full_theta = np.concatenate(([self.current_state[2]], ref_traj[2, :]))
        full_theta_unwrapped = np.unwrap(full_theta)
        ref_traj[2, :] = full_theta_unwrapped[1:]
        
        # 2. Giải MPC (Dùng lại Solver cũ)
        u_opt, x_pred = self.solver.solve(self.current_state, ref_traj)
        
        # 3. Gửi lệnh
        cmd = Twist()
        cmd.linear.x = float(u_opt[0])
        cmd.angular.z = float(u_opt[1])
        self.pub_cmd.publish(cmd)
        
        # 4. Vẽ
        self.visualize_path(ref_traj, self.pub_ref_path)
        self.visualize_path(x_pred, self.pub_pred_path)

def main(args=None):
    rclpy.init(args=args)
    node = MPCScenarioSquare()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
