import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import math

# Dùng Solver Cơ bản (Tracking)
from mpc_bot.mpc_solver import MPCSolver

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class ScenarioNav(Node):
    def __init__(self):
        super().__init__('scenario_nav')
        
        self.dt = 0.1
        self.N = 20
        self.solver = MPCSolver(self.N, self.dt)
        self.get_logger().info("📍 Kịch bản ĐI ĐẾN ĐIỂM ĐÍCH (Point-to-Point) đã sẵn sàng!")
        self.get_logger().info("👉 Hãy dùng công cụ '2D Goal Pose' trên Rviz để chọn điểm đến!")

        # Setup ROS
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Lắng nghe lệnh từ Rviz (Nút 2D Goal Pose)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_ref_path = self.create_publisher(Path, '/mpc/reference_path', 10)
        self.pub_pred_path = self.create_publisher(Path, '/mpc/predicted_path', 10)
        
        self.timer = self.create_timer(self.dt, self.control_loop)

        # Trạng thái
        self.current_state = np.array([0.0, 0.0, 0.0]) 
        self.got_odom = False
        
        # Biến quản lý mục tiêu
        self.has_goal = False
        self.goal_state = np.array([0.0, 0.0, 0.0]) # [x, y, theta]
        self.is_reached = False

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_state = np.array([x, y, theta])
        self.got_odom = True

    def goal_callback(self, msg):
        """Hàm này chạy khi bạn click chuột trên Rviz"""
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        
        # Lấy góc theta mục tiêu từ quaternion của chuột
        q = msg.pose.orientation
        gtheta = euler_from_quaternion(q.x, q.y, q.z, q.w)
        
        self.goal_state = np.array([gx, gy, gtheta])
        self.has_goal = True
        self.is_reached = False
        self.get_logger().info(f"🎯 Nhận mục tiêu mới: X={gx:.2f}, Y={gy:.2f}")

    def generate_path_to_goal(self):
        """
        Tạo quỹ đạo nội suy từ vị trí hiện tại đến đích.
        Đơn giản nhất: Tạo đường thẳng nối 2 điểm.
        """
        ref_matrix = np.zeros((3, self.N))
        
        # Vector từ xe đến đích
        dx = self.goal_state[0] - self.current_state[0]
        dy = self.goal_state[1] - self.current_state[1]
        dist_to_goal = math.sqrt(dx**2 + dy**2)
        
        # Nếu đã đến rất gần đích (< 10cm) -> Dừng lại
        if dist_to_goal < 0.1:
            self.is_reached = True
            # Tạo quỹ đạo đứng yên tại đích
            for i in range(self.N):
                ref_matrix[:, i] = self.goal_state
            return ref_matrix

        # Nếu chưa đến đích -> Tạo đường dẫn
        # Logic: Giả sử ta muốn đi đến đích với vận tốc v_desired
        v_des = 0.5
        
        # Góc hướng về đích
        angle_to_goal = math.atan2(dy, dx)
        
        for i in range(self.N):
            # Tính quãng đường dự kiến đi được sau i bước
            dist_future = v_des * (i * self.dt)
            
            # Nếu quãng đường này vượt quá đích -> Kẹp lại tại đích
            if dist_future > dist_to_goal:
                ref_matrix[0, i] = self.goal_state[0]
                ref_matrix[1, i] = self.goal_state[1]
                ref_matrix[2, i] = self.goal_state[2] # Hướng cuối cùng mong muốn
            else:
                # Nội suy tuyến tính (Linear Interpolation)
                ratio = dist_future / dist_to_goal
                ref_matrix[0, i] = self.current_state[0] + dx * ratio
                ref_matrix[1, i] = self.current_state[1] + dy * ratio
                
                # Hướng đi: Hướng về đích
                # Tuy nhiên, khi gần đến nơi, cần xoay xe về đúng hướng goal
                # Để đơn giản: Đoạn đầu hướng về đích, đoạn cuối xoay về goal_theta
                if dist_to_goal > 0.5:
                    ref_matrix[2, i] = angle_to_goal
                else:
                    # Gần đích thì nội suy góc xoay
                    ref_matrix[2, i] = self.goal_state[2]
                    
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
        if not self.got_odom:
            return

        if not self.has_goal:
            # Nếu chưa có goal, đứng yên hoặc giữ vị trí cũ
            # Gửi lệnh v=0
            self.pub_cmd.publish(Twist())
            return

        # 1. Tạo đường dẫn động (Dynamic Path Generation)
        ref_traj = self.generate_path_to_goal()
        
        # 2. Xử lý góc xoay (Unwrap)
        full_theta = np.concatenate(([self.current_state[2]], ref_traj[2, :]))
        full_theta_unwrapped = np.unwrap(full_theta)
        ref_traj[2, :] = full_theta_unwrapped[1:]
        
        # 3. Giải MPC
        u_opt, x_pred = self.solver.solve(self.current_state, ref_traj)
        
        # 4. Gửi lệnh (Nếu đã đến đích thì force stop)
        cmd = Twist()
        if self.is_reached:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("🏁 Đã đến đích!", once=True)
        else:
            cmd.linear.x = float(u_opt[0])
            cmd.angular.z = float(u_opt[1])
            
        self.pub_cmd.publish(cmd)
        
        # 5. Hiển thị
        self.visualize_path(ref_traj, self.pub_ref_path)
        self.visualize_path(x_pred, self.pub_pred_path)

def main(args=None):
    rclpy.init(args=args)
    node = ScenarioNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    
