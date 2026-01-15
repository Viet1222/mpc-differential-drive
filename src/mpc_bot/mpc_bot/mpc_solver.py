import casadi as ca
import numpy as np

# --- SOLVER CƠ BẢN (Dùng cho Hình 8, Hình Vuông) ---

def get_differential_drive_model(dt=0.1):
    x, y, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('theta')
    states = ca.vertcat(x, y, theta)
    v, omega = ca.SX.sym('v'), ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
    states_next = states + rhs * dt
    f = ca.Function('f', [states, controls], [states_next], ['state', 'control_input'], ['next_state'])
    return f, states.size1(), controls.size1()

class MPCSolver:
    def __init__(self, N=20, dt=0.1):
        self.N = N
        self.dt = dt
        f, n_states, n_controls = get_differential_drive_model(dt)
        self.n_states = n_states
        self.n_controls = n_controls
        
        # Trọng số
        Q = ca.diag([20.0, 20.0, 10.0]) 
        R = ca.diag([0.5, 0.5])
        S = ca.diag([10.0, 1.0])
        
        self.opti = ca.Opti()
        self.X = self.opti.variable(n_states, N + 1)
        self.U = self.opti.variable(n_controls, N)
        self.P_init = self.opti.parameter(n_states, 1)
        self.P_ref = self.opti.parameter(n_states, N)
        
        J = 0
        self.opti.subject_to(self.X[:, 0] == self.P_init)
        
        for k in range(N):
            st_err = self.X[:, k] - self.P_ref[:, k]
            J += ca.mtimes([st_err.T, Q, st_err])
            con = self.U[:, k]
            J += ca.mtimes([con.T, R, con])
            if k > 0:
                delta_U = self.U[:, k] - self.U[:, k-1]
                J += ca.mtimes([delta_U.T, S, delta_U])
            state_next = f(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == state_next)

        v_max, omega_max = 0.8, 1.0 
        self.opti.subject_to(self.opti.bounded(0.0, self.U[0, :], v_max))
        self.opti.subject_to(self.opti.bounded(-omega_max, self.U[1, :], omega_max))

        self.opti.minimize(J)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 50}
        self.opti.solver('ipopt', opts)

    def solve(self, current_state, reference_traj):
      
        self.opti.set_value(self.P_init, current_state)
        self.opti.set_value(self.P_ref, reference_traj)
        try:
            sol = self.opti.solve()
            return sol.value(self.U)[:, 0], sol.value(self.X)
        except:
            return np.zeros(self.n_controls), np.zeros((self.n_states, self.N + 1))
