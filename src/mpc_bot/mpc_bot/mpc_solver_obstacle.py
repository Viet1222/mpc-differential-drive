import casadi as ca
import numpy as np

def get_differential_drive_model(dt=0.1):
    x, y, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('theta')
    states = ca.vertcat(x, y, theta)
    v, omega = ca.SX.sym('v'), ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
    states_next = states + rhs * dt
    f = ca.Function('f', [states, controls], [states_next], ['state', 'control_input'], ['next_state'])
    return f, states.size1(), controls.size1()

class MPCObstacleSolver:
    def __init__(self, N=40, dt=0.1): # N=30 là hợp lý
        self.N = N
        self.dt = dt
        f, n_states, n_controls = get_differential_drive_model(dt)
        self.n_states = n_states
        self.n_controls = n_controls
        
        # --- BỘ TRỌNG SỐ ỔN ĐỊNH ---
        # Q: Bám đường
        Q = ca.diag([20.0, 20.0, 5.0]) 
        # R: Phạt năng lượng (vừa phải để không quá lì)
        R = ca.diag([1.0, 1.0]) 
        # S: Làm trơn (Quan trọng)
        S = ca.diag([5.0, 10.0])
        
        # Trọng số phạt Slack (Càng lớn thì càng sợ va chạm)
        # Đây là "độ cứng" của lò xo ảo khi va chạm
        W_SLACK = 10000.0 
        
        self.opti = ca.Opti()
        self.X = self.opti.variable(n_states, N + 1)
        self.U = self.opti.variable(n_controls, N)
        
        # --- 1. KHAI BÁO BIẾN SLACK ---
        # Biến này đại diện cho mức độ vi phạm vùng an toàn (nếu có)
        self.Slacks = self.opti.variable(N) 
        
        self.P_init = self.opti.parameter(n_states, 1)
        self.P_ref = self.opti.parameter(n_states, N)
        self.P_obs = self.opti.parameter(3, 1) 
        
        J = 0
        self.opti.subject_to(self.X[:, 0] == self.P_init)
        
        for k in range(N):
            # Tracking Cost
            st_err = self.X[:, k] - self.P_ref[:, k]
            J += ca.mtimes([st_err.T, Q, st_err])
            
            # Control Effort
            con = self.U[:, k]
            J += ca.mtimes([con.T, R, con])
            
            # Smoothness
            if k > 0:
                delta_U = self.U[:, k] - self.U[:, k-1]
                J += ca.mtimes([delta_U.T, S, delta_U])
            
            # Dynamics
            state_next = f(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == state_next)
            
            # --- 2. XỬ LÝ VẬT CẢN BẰNG SLACK VARIABLES ---
            # Khoảng cách bình phương
            dist_sq = (self.X[0, k] - self.P_obs[0])**2 + (self.X[1, k] - self.P_obs[1])**2
            
            # Bán kính an toàn bình phương (R_vat + R_xe + Margin)
            # Ví dụ: 0.4 + 0.2 + 0.1 = 0.7m
            safe_dist = self.P_obs[2] + 0.15 
            safe_sq = safe_dist**2
            
            # Ràng buộc: Khoảng cách + Slack >= An toàn
            # Nếu dist_sq < safe_sq, Slack buộc phải dương để bù vào -> Bị phạt nặng
            self.opti.subject_to(dist_sq + self.Slacks[k] >= safe_sq)
            
            # Ràng buộc Slack không âm
            self.opti.subject_to(self.Slacks[k] >= 0.0)
            
            # Thêm phạt Slack vào hàm mục tiêu (L2 norm cho mượt gradient)
            J += W_SLACK * (self.Slacks[k]**2)

        # Ràng buộc vật lý
        v_max, omega_max = 0.8, 1.0 
        self.opti.subject_to(self.opti.bounded(0.0, self.U[0, :], v_max))
        self.opti.subject_to(self.opti.bounded(-omega_max, self.U[1, :], omega_max))

        self.opti.minimize(J)
        # Tăng max_iter lên một chút để đảm bảo hội tụ
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100, 'ipopt.warm_start_init_point': 'yes'}
        self.opti.solver('ipopt', opts)

    def solve(self, current_state, reference_traj, obstacle_params):
        self.opti.set_value(self.P_init, current_state)
        self.opti.set_value(self.P_ref, reference_traj)
        self.opti.set_value(self.P_obs, obstacle_params)
        
        # Warm start (Tùy chọn, giúp giải nhanh hơn)
        # self.opti.set_initial(self.U, ...) 
        
        try:
            sol = self.opti.solve()
            # Lấy giá trị cost (nếu cần vẽ đồ thị)
            cost_val = sol.value(self.opti.f)
            return sol.value(self.U)[:, 0], sol.value(self.X) #, cost_val
        except:
            # Fallback an toàn
            return np.array([0.0, 0.0]), np.zeros((self.n_states, self.N + 1)) #, 0.0
