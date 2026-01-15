import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
import numpy as np
import math
import csv
import os

# Dùng Solver NÂNG CAO
from mpc_bot.mpc_solver_obstacle import MPCObstacleSolver

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class ScenarioObstacle(Node):
    def __init__(self):
        super().__init__('scenario_obstacle')
        
        self.dt = 0.1
        self.N = 40
        self.solver = MPCObstacleSolver(self.N, self.dt)
        self.get_logger().info("🚧 Kịch bản Tránh Vật Cản (Kèm Ghi Log) đã chạy!")
        
        # --- LOGGING ---
        self.csv_file_path = os.path.expanduser('~/mpc_data_obstacle.csv')
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        header = ['time', 'x_ref', 'y_ref', 'theta_ref', 'x_act', 'y_act', 'theta_act', 'v_cmd', 'omega_cmd']
        self.csv_writer.writerow(header)
        
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
        
        self.obs_x = 2.5
        self.obs_y = 0.05
        self.obs_r = 0.5
        self.obs_param = np.array([self.obs_x, self.obs_y, self.obs_r + 0.3])

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_state = np.array([x, y, theta])
        self.got_odom = True

    def generate_straight_line(self):
        t_now = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        ref_matrix = np.zeros((3, self.N))
        v_ref = 0.5
        for i in range(self.N):
            dt_step = t_now + i * self.dt
            x_ref = v_ref * dt_step
            if x_ref > 6.0: x_ref = 6.0
            ref_matrix[0, i] = x_ref
            ref_matrix[1, i] = 0.0
            ref_matrix[2, i] = 0.0
        return ref_matrix

    def publish_obstacle_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.obs_x
        marker.pose.position.y = self.obs_y
        marker.scale.x = (self.obs_r + 0.3) * 2
        marker.scale.y = (self.obs_r + 0.3) * 2
        marker.scale.z = (self.obs_r + 0.3) * 2
        marker.color.a = 0.5; marker.color.r = 1.0; 
        self.pub_marker.publish(marker)
        
    def visualize_path(self, trajectory_matrix, publisher):
        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(trajectory_matrix.shape[1]):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(trajectory_matrix[0, i])
            pose.pose.position.y = float(trajectory_matrix[1, i])
            path_msg.poses.append(pose)
        publisher.publish(path_msg)

    def control_loop(self):
        if not self.got_odom: return
        ref_traj = self.generate_straight_line()
        
        u_opt, x_pred = self.solver.solve(self.current_state, ref_traj, self.obs_param)
        
        v_cmd = float(u_opt[0])
        w_cmd = float(u_opt[1])
        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.pub_cmd.publish(cmd)
        
        self.visualize_path(ref_traj, self.pub_ref_path)
        self.visualize_path(x_pred, self.pub_pred_path)
        self.publish_obstacle_marker()
        
        current_time = self.get_clock().now().nanoseconds / 1e9 - self.initial_time
        self.csv_writer.writerow([
            f"{current_time:.2f}",
            f"{ref_traj[0,0]:.3f}", f"{ref_traj[1,0]:.3f}", f"{ref_traj[2,0]:.3f}",
            f"{self.current_state[0]:.3f}", f"{self.current_state[1]:.3f}", f"{self.current_state[2]:.3f}",
            f"{v_cmd:.3f}", f"{w_cmd:.3f}"
        ])

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ScenarioObstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
