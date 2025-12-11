import numpy as np
import casadi as ca
from scipy.spatial import ConvexHull


class EPSF_Controller:

    def __init__(self,
                dynamics,
                horizon : int,
                control_frequency : float,
                neural_model,
                bounds : dict,
                obstacles = None,
                backup_policy = None,
                R = np.diag([1.0, 0.1]),
                Q = np.diag([1.0, 1.0, 1.0, 0.0, 0.0]),
                diff_R = np.diag([0.1, 0.1]),
                terminal_ingredients=None):


        self._N = horizon
        self._dt = 1 / control_frequency

        self._bounds = bounds
        self._Q = Q
        self._n_x = 5
        self._n_u = 2
        self._R = R
        self._diff_R = diff_R

        self._dynamics = dynamics

        self._neural_model = neural_model

        self._terminal_ingredients = terminal_ingredients
        self._pose = None
        self._velocities = None
        self._target_pose = None
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

            pose_x, pose_y, pose_yaw = state[0:3]
            pose_yaw = (pose_yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            lin_vel, ang_vel = self.get_velocities_from_cmd(state[3:5])
            current_gamma = self._gamma

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
        for i in range(self._N):
            v_cmd, gamma_action = action[2*i], action[2*i + 1]
            x_next = self.rk4_step(x_init, action)
            x_next[2] =  ca.fmod(x_next[2] + ca.pi, 2 * ca.pi) - ca.pi
            x_states.append(x_next)
            x_init = x_next

        return np.array(x_states)


    
    def provide_weight_from_geodesic_dist(self, x, x_rl) :

        psf_state = x[0:3]
        rl_state = x_rl[0:3]
        dist = np.sqrt((psf_state[0] - rl_state[0])**2 + (psf_state[1] - rl_state[1])**2 + (psf_state[2] - rl_state[2])**2)

        return np.exp(-dist)



    def obstacle_to_ellipse(self, points: np.ndarray, safety_radius: float = 1.0):

        assert points.shape[1] == 2, "Les points doivent être en 2D"

        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]

        center = np.mean(hull_pts, axis=0)

        distances = np.linalg.norm(hull_pts - center, axis=1)
        max_dist = np.max(distances)

        radius = max_dist + safety_radius

        return {
            'center': center,
            'radius': radius
        }



    def build_nlp(self, x_init, x_ref, u_ref, h):

        nx, nu = self._n_x, self._n_u
        X = ca.MX.sym('X', nx, h+1)
        U = ca.MX.sym('U', nu, h)

        cost = 0
        g_ineq = []
        g_equalities = []

        weight = 1
        # #penalize variations in control
        for k in range(h):

            u_ref_k = np.array([u_ref[2*k], u_ref[2*k + 1]]).T
            diff_cmd = ca.DM(u_ref_k) - U[:, k]

            x_ref_k = np.array(x_ref[k])
            state_error = ca.DM(x_ref_k) - X[:, k+1]

            weight *= self.provide_weight_from_geodesic_dist(X[:, k+1], x_ref[k])

            cost += weight * (diff_cmd.T @ self._R @ diff_cmd)
            cost += (1-weight) * ((state_error.T @ self._Q @ state_error)) #+ U[:, k].T @ self._diff_R @ U[:, k])

        for k in range(1, h):

            diff = U[:,k] - U[:,k-1] 
            cost += ca.mtimes([diff.T, 0.5 * ca.DM(self._diff_R), diff])

        # inequality constraints : bornes
        u_min = self._bounds["u_min"]
        u_max = self._bounds["u_max"]
        x_min = self._bounds["x_min"]
        x_max = self._bounds["x_max"]

        for k in range(h):
            x_next_pred = self.rk4_step(X[:, k], U[:, k])
            x_next_pred[2] = ca.fmod(x_next_pred[2] + ca.pi, 2 * ca.pi) - ca.pi
            g_equalities.append(X[:, k+1] - x_next_pred)

            for i in range(nx):
                g_ineq += [x_min[i] - X[i, k]]
                g_ineq += [X[i, k] - x_max[i]]

            for i in range(nu):
                g_ineq += [u_min[i] - U[i, k]]
                g_ineq += [U[i, k] - u_max[i]]


        for obstacle in self.obstacles:
            ellipse = self.obstacle_to_ellipse(obstacle, safety_radius=1.0)
            center = ellipse['center']
            radius = ellipse['radius']

            for k in range(h):
                dist_sq = (X[0, k] - center[0])**2 + (X[1, k] - center[1])**2
                g_ineq += [radius**2 - dist_sq]

            dist_sq_final = (X[0, h] - center[0])**2 + (X[1, h] - center[1])**2
            g_ineq += [radius**2 - dist_sq_final]

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
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        g_eq = sizes['g_eq_len']
        g_ineq = sizes['g_ineq_len']
        horizon = sizes['horizon']

        lbg = [0]*g_eq + [-ca.inf]*g_ineq
        ubg = [0]*g_eq + [0]*g_ineq

        # melt x_init and x_rl into initial guess
        # X0 = np.zeros((self._n_x, horizon+1))
        # X0[:,0] = np.array(x_init).T
        # for k in range(horizon):
        #     x_ref_k = np.array(x_ref[k]).T
        #     X0[:,k+1] = x_ref_k

        X0 = np.tile(x_init.reshape(-1,1), (horizon+1))
    
        U0 = np.zeros((self._n_u, horizon))
        for k in range(horizon):
            u_ref_k = np.array([u_ref[2*k], u_ref[2*k + 1]]).T
            U0[:,k] = u_ref_k
    
        x_init_tile = X0.flatten(order='F')
        u_init_tile = U0.flatten(order='F')

        initial_guess = np.concatenate([x_init_tile, u_init_tile])

        sol = solver(x0=initial_guess, lbg=lbg, ubg=ubg)

        stats = solver.stats()
        if not stats['success']:
            print("NLP solver failed")
            return None, None   

        else : 
            w_opt = np.array(sol['x']).squeeze()
            X_opt = w_opt[:self._n_x*(horizon+1)].reshape((self._n_x, horizon+1), order='F')
            U_opt = w_opt[self._n_x*(horizon+1):self._n_x*(horizon+1)+self._n_u*horizon].reshape((self._n_u, horizon), order='F')

            return X_opt, U_opt


    
    def solve_global_problem(self, x_init, u_rl, x_rl) :

        #feasible = False
        # X_opt = None
        # U_opt = None

        nlp, sizes, g_equalities, g_ineq = self.build_nlp(x_init, x_rl, u_rl, self._N)
        X_opt, U_opt = self.solve_nlp(nlp, sizes, u_rl, x_init, x_rl)

        if X_opt is not None and U_opt is not None:
            self._kbar = self._k
            return X_opt, U_opt

        else : 
            if self._k < self._kbar + self._N:
                dt = self._k - self._kbar
                reduced_horizon = self._N - dt
                x_rl = x_rl[:reduced_horizon]
                u_rl = u_rl[:2*reduced_horizon + 1]

                nlp_red, sizes_red, g_eq_red, g_ineq_red = self.build_nlp(
                    x_init, x_rl,  u_rl, reduced_horizon)
                X_red, U_red = self.solve_nlp(nlp_red, sizes_red, u_rl, x_init, x_rl)

                if X_red is not None and U_red is not None:
                    return X_red, U_red

        X_opt = x_init.reshape(-1, 1).repeat(self._N, axis=1)
        # if self.backup_policy is not None :
        #     U_opt = self.backup_policy.act(x_init)
        #     U_opt = U_opt.reshape(-1, 1)
        # else :
        u_opt = [0.0, 0.0]
        U_opt = np.array(u_opt).reshape(-1, 1).repeat(self._N, axis=1)

        return X_opt, U_opt



    def get_velocities_from_cmd(self, u_cmd) :
        v_ref, gamma = u_cmd
        return np.array([v_ref, 2 * v_ref * np.sin(gamma / 2) / 4])



    def act(self, x_init) :

        obs = self._compute_agent_vect(target_states=self._target_vector, state=x_init)
        u_rl = self.get_neural_prediction_actions(obs)[0]
        x_rl = self.get_neural_prediction_states(u_rl, x_init)
        #x_interm = x_init.copy()

        X_opt, U_opt = self.solve_global_problem(x_init, u_rl, x_rl)
        U_opt_list = []
        for i in range(U_opt.shape[1]):
            U_opt_list.append(U_opt[0, i])
            U_opt_list.append(U_opt[1, i])
        opt_cmd = U_opt[:, 0]

        return opt_cmd, X_opt, u_rl, U_opt_list