import numpy as np
import casadi as ca
from scipy.spatial import ConvexHull
from stable_baselines3 import PPO



class EPSF_Controller:

    def __init__(self,
                dynamics,
                horizon : int,
                control_frequency : float,
                neural_model_path,
                bounds : dict,
                obstacles = None,
                backup_policy = None,
                R = np.diag([1.0, 0.1]),
                Q = np.diag([1.0, 1.0, 1.0, 0.0, 0.0]),
                Q_t = np.diag([1.0, 1.0, 1.0]),
                diff_R = np.diag([0.0, 1.0]),
                PR = np.diag([0.0, 0.0]),
                terminal_ingredients=None):


        self._N = horizon
        self._dt = 1 / control_frequency

        self._bounds = bounds
        self._Q = Q
        self._Q_t = Q_t
        self._n_x = 7
        self._n_u = 2
        self._R = R
        self._diff_R = diff_R
        self._PR = PR
        self._Jb = 0.

        self._dynamics = dynamics

        self._neural_model_path = neural_model_path
        self._neural_model = PPO.load(neural_model_path)

        self._terminal_ingredients = terminal_ingredients
        self._pose = None
        self._velocities = None
        self._target_pose = None
        self._gamma = 0.
        self._target_velocities = None
        self.neural_states = None

        self.episode_start = True

        self.obstacles = obstacles if obstacles is not None else []

        self.backup_policy = backup_policy
        self._kbar = 0
        self._k = 0
        self._X_plan = None
        self._U_plan = None
        self._prev_cmd = None

        rho = 0.5
        self.epsilon = 0.9 * rho



    def get_parameters(self):
        return {
            "horizon": self._N,
            "neural_model_path": self._neural_model_path,
            "control_frequency": 1 / self._dt,
            "R": self._R,
            "Q": self._Q,
            "diff_R": self._diff_R,
            "PR": self._PR,
        }


    def set_target_velocities(self, target_velocities):
        self._target_velocities = target_velocities

    def set_target_pose(self, target_pose):
        self._target_pose = target_pose

    def set_target_vector(self, target_vector):
        self._target_vector = target_vector

    def set_current_pose(self, pose):
        self._pose = pose

    def set_current_velocities(self, velocities):
        self._velocities = velocities

    def set_gamma(self, gamma):
        self._gamma = gamma

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles



    def rk4_step(self, x, u):
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + (self._dt/2)*k1, u)
        k3 = self._dynamics(x + (self._dt/2)*k2, u)
        k4 = self._dynamics(x + self._dt*k3, u)
        return x + (self._dt/6)*(k1 + 2*k2 + 2*k3 + k4)


    def update_Jb_from_solution(self, x_k, X_opt, cost):

        epsilon = 0.05

        e0 = x_k[0:5] - self._target_vector[:, 0]
        ell_k = float(e0.T @ self._Q_t @ e0)

        if cost is None :
            return 

        self._Jb = cost - epsilon * ell_k

        print(f"Updated Jb to {self._Jb:.3f}")
        return



    def initialize_Jb(self, x_init):

        nlp, sizes, _, _ = self.build_mpc_classic_pb(x_init)
        X_opt, U_opt, V_cost = self.solve_mpc(nlp, sizes, x_init)

        if X_opt is None:
            raise RuntimeError("Failed to initialize Jb")

        self._Jb = 8 * float(V_cost)

        print(f"Jb initialized to {self._Jb:.3f}")

        
    def build_mpc_classic_pb(self, x_init):

        nx, nu = self._n_x, self._n_u
        X = ca.MX.sym('X', nx, self._N + 1)
        U = ca.MX.sym('U', nu, self._N)

        J_cost = 0
        g_ineq = []
        g_eq = []

        Q = ca.MX(self._Q_t)

        for k in range(self._N):
            target_k = ca.MX(self._target_vector[:, k])
            e = X[0:5, k] - target_k
            J_cost += ca.mtimes([e.T, Q, e])

        # Dynamics and constraints
        u_min, u_max = self._bounds["u_min"], self._bounds["u_max"]
        x_min, x_max = self._bounds["x_min"], self._bounds["x_max"]

        for k in range(self._N):
            x_next = self.rk4_step(X[:, k], U[:, k])
            #x_next[4] = ca.fmod(x_next[4] + ca.pi, 2 * ca.pi) - ca.pi
            g_eq.append(X[:,k+1] - x_next)

        for k in range(self._N + 1):
            for i in range(nx):
                g_ineq += [x_min[i] - X[i, k], X[i, k] - x_max[i]]

        for k in range(self._N):
            for i in range(nu):
                g_ineq += [u_min[i] - U[i, k], U[i, k] - u_max[i]]

        # Obstacles
        for obstacle in self.obstacles:

            x_min = obstacle['x_min']
            x_max = obstacle['x_max']
            y_min = obstacle['y_min']
            y_max = obstacle['y_max']

            d_safe = 1.

            for k in range(self._N+1):
                x = X[2, k]
                y = X[3, k]

                c1 = x_min - x - d_safe
                c2 = x - x_max - d_safe
                c3 = y_min - y - d_safe
                c4 = y - y_max - d_safe

                outside = self.smooth_max(ca.vertcat(c1, c2, c3, c4))
                g_ineq += [-outside]  # outside >= 0


        g_eq.append(X[:, 0] - x_init)

        g_eq = ca.vertcat(*g_eq)
        g_ineq = ca.vertcat(*g_ineq)

        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1)
        )

        nlp = {
            'x': opt_vars,
            'f': J_cost,
            'g': ca.vertcat(g_eq, g_ineq)
        }

        sizes = {
            'nx_block': nx * (self._N + 1),
            'nu_block': nu * self._N,
            'g_eq_len': g_eq.numel(),
            'g_ineq_len': g_ineq.numel(),
            'horizon': self._N
        }

        return nlp, sizes, g_eq, g_ineq



    def solve_mpc(self, nlp, sizes, x_init) : 

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        g_eq = sizes['g_eq_len']
        g_ineq = sizes['g_ineq_len']
        horizon = sizes['horizon']

        lbg = [0]*g_eq + [-ca.inf]*g_ineq
        ubg = [0]*g_eq + [0.0001]*g_ineq

        # melt x_init and x_rl into initial guess
        X0 = np.zeros((self._n_x, horizon+1))
        X0[:,0] = np.array(x_init).T

        X0 = np.tile(x_init.reshape(-1,1), (horizon+1))
    
        U0 = np.zeros((self._n_u, horizon))

    
        x_init_tile = X0.flatten(order='F')
        u_init_tile = U0.flatten(order='F')

        initial_guess = np.concatenate([x_init_tile, u_init_tile])

        sol = solver(x0=initial_guess, lbg=lbg, ubg=ubg)

        stats = solver.stats()
        if not stats['success']:
            print("MPC solver failed")
            return None, None, None  

        else : 
            w_opt = np.array(sol['x']).squeeze()
            X_opt = w_opt[:self._n_x*(horizon+1)].reshape((self._n_x, horizon+1), order='F')
            U_opt = w_opt[self._n_x*(horizon+1):self._n_x*(horizon+1)+self._n_u*horizon].reshape((self._n_u, horizon), order='F')
            V = sol['f']

            return X_opt, U_opt, V

        
        
    def get_neural_prediction_actions(self, obs) :

        obs = np.array(obs).reshape(1, -1)
        action, _ = self._neural_model.predict(
            obs,
            deterministic=True
        )
        self.episode_start = False
        return action.tolist()



    def _compute_agent_vect(self, target_states, state):
        """
        Returns a vector describing the agent state relative to the *entire target sequence*:
        - forward and lateral error for each target
        - orientation error for each target
        - error in forward speed
        - error in angular velocity
        """

        agent_state_vect = []

        # Iterate over the target sequence
        for i in range(self._N):
            target_state = target_states[:, i]  # assuming you store the whole sequence in self._target_sequence
            target_lin_vel, target_ang_vel = target_state[0:2]
            target_x, target_y, target_yaw = target_state[2:5]

            pose_x, pose_y, pose_yaw = state[2:5]
            pose_yaw = (pose_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            lin_vel, ang_vel = state[0:2]
            current_gamma = state[5]

            dx = target_x - pose_x
            dy = target_y - pose_y

            # Forward/lateral error in agent frame
            forward_error = np.cos(pose_yaw) * dx + np.sin(pose_yaw) * dy
            lateral_error = -np.sin(pose_yaw) * dx + np.cos(pose_yaw) * dy

            # Orientation error
            delta_yaw = (target_yaw - pose_yaw + np.pi) % (2 * np.pi) - np.pi

            # Velocity error in robot frame
            target_vel_robot_frame = np.array([
                target_lin_vel * np.cos(target_yaw - pose_yaw),
                target_lin_vel * np.sin(target_yaw - pose_yaw)
            ])
            vel_error_x = target_vel_robot_frame[0] - lin_vel
            ang_vel_error = target_ang_vel - ang_vel

            # Gamma error
            if np.abs(target_ang_vel) < 1e-2 or np.abs(target_lin_vel) < 1e-2:
                target_gamma = 0.0
            else:
                ratio = np.clip((4 * target_ang_vel) / (2 * target_lin_vel), -1.0, 1.0)
                target_gamma = 2 * np.arcsin(ratio)

            gamma_error = (target_gamma - current_gamma + np.pi) % (2 * np.pi) - np.pi

            # Append all errors for this target to the vector
            agent_state_vect.extend([
                forward_error,
                lateral_error,
                delta_yaw,
                vel_error_x,
                gamma_error,
            ])

        return np.array(agent_state_vect)       
        


    def get_neural_prediction_actions(self, obs) :

        obs = np.array(obs).reshape(1, -1)
        action, _ = self._neural_model.predict(
            obs,
            deterministic=True
        )
        self.episode_start = False
        return action.tolist()



    def get_neural_prediction_states(self, action, x_init) :

        # use dynamics to predict next state
        x_states = []
        x_states.append(x_init)
        for i in range(self._N):
            v_cmd, gamma_action = action[2*i], action[2*i + 1]
            x_next = self.rk4_step(x_init, action)
            x_next[4] =  ca.fmod(x_next[4] + ca.pi, 2 * ca.pi) - ca.pi
            x_states.append(x_next)
            x_init = x_next

        return x_states


    def provide_weight_from_geodesic_dist(self, x, x_rl) :

        psf_state = x[0:5]
        rl_state = x_rl[0:5]
        dist = np.sqrt((psf_state[3] - rl_state[3])**2 + (psf_state[4] - rl_state[4])**2 + (psf_state[2] - rl_state[2])**2)

        return np.exp(-dist)



    def smooth_max(self, v, alpha=10.0):
        return (1/alpha) * ca.log(ca.sum1(ca.exp(alpha * v)))


    def build_nlp(self, x_init, x_ref, u_ref, h):

        nx, nu = self._n_x, self._n_u
        X = ca.MX.sym('X', nx, h+1)
        U = ca.MX.sym('U', nu, h)

        cost = 0
        g_ineq = []
        g_equalities = []

        weight = 1.0
        J_cost = 0

        for k in range(h):

            x_ref_k = np.array(x_ref[k+1])
            state_error = ca.DM(x_ref_k) - X[:, k+1]

            u_ref_k = np.array([u_ref[2*k], u_ref[2*k + 1]]).T
            diff_cmd = ca.DM(u_ref_k) - U[:, k]

            weight *= self.provide_weight_from_geodesic_dist(X[:, k+1], x_ref[k+1])

            #cost += (1-weight) * (state_error.T @ self._Q @ state_error)
            cost += weight * (diff_cmd.T @ self._R @ diff_cmd)


        for k in range(h):
            
            target_k = self._target_vector[:, k]
            psf_state_k = X[:, k][0:5]
            error = ca.DM(target_k) - psf_state_k
            J_cost += ca.mtimes([error.T, self._Q_t, error])


        g_ineq += [J_cost - self._Jb]

        # inequality constraints : bornes
        u_min = self._bounds["u_min"]
        u_max = self._bounds["u_max"]
        x_min = self._bounds["x_min"]
        x_max = self._bounds["x_max"]

        for k in range(h):
            x_next_pred = self.rk4_step(X[:, k], U[:, k])
            x_next_pred[4] = ca.fmod(x_next_pred[4] + ca.pi, 2 * ca.pi) - ca.pi
            g_equalities.append(X[:, k+1] - x_next_pred)

        for k in range(h+1):
            for i in range(nx):
                g_ineq += [x_min[i] - X[i, k]]
                g_ineq += [X[i, k] - x_max[i]]

        for k in range(h):

            for i in range(nu):
                g_ineq += [u_min[i] - U[i, k]]
                g_ineq += [U[i, k] - u_max[i]]

        d_safe = 1.

        # for obstacle in self.obstacles:
            
        #     x_min = obstacle['x_min']
        #     x_max = obstacle['x_max']
        #     y_min = obstacle['y_min']
        #     y_max = obstacle['y_max']

        #     for k in range(h):
        #         x = X[2, k]
        #         y = X[3, k]

        #         c1 = x_min - x - d_safe
        #         c2 = x - x_max - d_safe
        #         c3 = y_min - y - d_safe
        #         c4 = y - y_max - d_safe

        #         outside = self.smooth_max(ca.vertcat(c1, c2, c3, c4))
        #         g_ineq += [-outside]  # outside >= 0

        #     x = X[2, h]
        #     y = X[3, h]

        #     c1 = x_min - x - (d_safe * 2)
        #     c2 = x - x_max - (d_safe * 2)
        #     c3 = y_min - y - (d_safe * 2)
        #     c4 = y - y_max - (d_safe * 2)

        #     outside = self.smooth_max(ca.vertcat(c1, c2, c3, c4))
        #     g_ineq += [-outside]  # outside >= 0
            

        g_equalities.append(X[:,0] - x_init)
        g_equalities = ca.vertcat(*g_equalities)
        g_ineq = ca.vertcat(*g_ineq)

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1),
                              ca.reshape(U, -1, 1))

        nlp = {'x': opt_vars, 'f': cost, 'g': ca.vertcat(g_equalities, g_ineq)}

        sizes = {
            'nx_block': nx*(h+1),
            'nu_block': nu*h,
            'g_eq_len': g_equalities.numel(),
            'g_ineq_len': g_ineq.numel(),
            'horizon': h
        }

        return nlp, sizes, g_equalities, g_ineq



    def solve_nlp(self, nlp, sizes, u_ref, x_init, x_ref) : 

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_cpu_time": 1.5
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        g_eq = sizes['g_eq_len']
        g_ineq = sizes['g_ineq_len']
        horizon = sizes['horizon']

        lbg = [0]*g_eq + [-ca.inf]*g_ineq
        ubg = [0]*g_eq + [0.00001]*g_ineq

        # melt x_init and x_rl into initial guess
        # X0 = np.zeros((self._n_x, horizon+1))
        # for k in range(horizon):
        #     x_ref_k = np.array(x_ref[k]).T
        #     X0[:, k] = x_ref_k

        U0 = np.zeros((self._n_u, horizon))
        for k in range(horizon):
            u_ref_k = np.array([u_ref[2*k], u_ref[2*k + 1]]).T
            U0[:,k] = u_ref_k

        X0 = np.zeros((self._n_x, horizon+1))
        X0[:, 0] = x_init

        for k in range(horizon):
            X0[:, k+1] = np.array(
                self.rk4_step(
                    ca.DM(X0[:, k]),
                    ca.DM(U0[:, k])
                )
            ).squeeze()

    
        x_init_tile = X0.flatten(order='F')
        u_init_tile = U0.flatten(order='F')

        initial_guess = np.concatenate([x_init_tile, u_init_tile])

        sol = solver(x0=initial_guess, lbg=lbg, ubg=ubg)

        stats = solver.stats()
        if not stats['success']:
            print("NLP solver failed")
            return None, None, None   

        else : 
            w_opt = np.array(sol['x']).squeeze()
            X_opt = w_opt[:self._n_x*(horizon+1)].reshape((self._n_x, horizon+1), order='F')
            U_opt = w_opt[self._n_x*(horizon+1):self._n_x*(horizon+1)+self._n_u*horizon].reshape((self._n_u, horizon), order='F')
            J_cost = 0.
            for k in range(horizon):
                diff_x = X_opt[:, k][0:5] - self._target_vector[:, k]
                X_opt[5, k] = (X_opt[5, k] + np.pi) % (2 * np.pi) - np.pi
                J_cost += float(diff_x.T @ self._Q_t @ diff_x)
            print(f"Solved NLP with cost {J_cost:.3f}")
            return X_opt, U_opt, J_cost


    
    def solve_global_problem(self, x_init, u_rl, x_rl) :

        #feasible = False
        # X_opt = None
        # U_opt = None

        need_init = False

        nlp, sizes, g_equalities, g_ineq = self.build_nlp(x_init, x_rl, u_rl, self._N)
        X_opt, U_opt, cost = self.solve_nlp(nlp, sizes, u_rl, x_init, x_rl)

        if X_opt is not None and U_opt is not None:
            self._kbar = self._k
            return X_opt, U_opt, cost, need_init

        # nlp, sizes, g_equalities, g_ineq = self.build_nlp_obstacle(x_init, x_rl, u_rl, self._N)
        # X_opt, U_opt, cost = self.solve_nlp_obstacle(nlp, sizes, u_rl, x_init, x_rl)
        # need_init = True

        # if X_opt is not None and U_opt is not None:
        #     self._kbar = self._k
        #     return X_opt, U_opt, cost, need_init

        else : 
            if self._k < self._kbar + self._N:
                dt = self._k - self._kbar
                reduced_horizon = self._N - dt
                x_rl = x_rl[:reduced_horizon+1]
                u_rl = u_rl[:2*reduced_horizon + 1]

                nlp_red, sizes_red, g_eq_red, g_ineq_red = self.build_nlp(
                    x_init, x_rl,  u_rl, reduced_horizon)
                X_red, U_red, cost_red = self.solve_nlp(nlp_red, sizes_red, u_rl, x_init, x_rl)

                if X_red is not None and U_red is not None:
                    return X_red, U_red, cost_red, need_init

        X_fallback = x_init.reshape(-1, 1).repeat(self._N, axis=1)

        if self.backup_policy is not None:
            U_fallback = self.backup_policy.act(x_init)
            U_fallback = U_fallback.reshape(-1, 1).repeat(self._N, axis=1)
        else:
            x = x_init[2]
            y = x_init[3]
            target_x = self._target_vector[2, 0]
            target_y = self._target_vector[3, 0]
            pose_yaw = x_init[4]
            dx = target_x - x
            dy = target_y - y

            # Forward/lateral error in agent frame
            forward_error = np.cos(pose_yaw) * dx + np.sin(pose_yaw) * dy
            lateral_error = -np.sin(pose_yaw) * dx + np.cos(pose_yaw) * dy

            kp_v = 0.02
            kp_gamma = 0.01
            v_cmd = kp_v * forward_error
            gamma_cmd = kp_gamma * np.arctan2(lateral_error, forward_error)
            U_fallback = np.array([v_cmd, gamma_cmd]).reshape(-1, 1).repeat(self._N, axis=1)

        return X_fallback, U_fallback, cost_red, need_init


    def get_velocities_from_cmd(self, u_cmd) :
        v_ref, gamma = u_cmd
        return np.array([v_ref, 2 * v_ref * np.sin(gamma / 2) / 4])



    def act(self, x_init) :

        obs = self._compute_agent_vect(target_states=self._target_vector, state=x_init)
        u_rl = self.get_neural_prediction_actions(obs)[0]
        x_rl = self.get_neural_prediction_states(u_rl, x_init)
        #x_interm = x_init.copy()

        X_opt, U_opt, cost_red, need_init = self.solve_global_problem(x_init, u_rl, x_rl)
        U_opt_list = []
        for i in range(U_opt.shape[1]):
            U_opt_list.append(U_opt[0, i])
            U_opt_list.append(U_opt[1, i])
        opt_cmd = U_opt[:, 0]

        return opt_cmd, X_opt, u_rl, U_opt_list, cost_red