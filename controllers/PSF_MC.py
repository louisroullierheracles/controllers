import numpy as np
import casadi as ca


class PSF_MC_Controller:

    def __init__(self,
                dynamics,
                horizon : int,
                control_frequency : float,
                neural_model,
                bounds : dict,
                R: np.ndarray,
                terminal_ingredients=None):

        self._N = horizon
        self._dt = 1 / control_frequency

        self._R = R

        self._bounds = bounds

        self._n_x = 4
        self._n_u = 2

        self.gamma = 0.95

        self._dynamics = dynamics

        self._neural_model = neural_model

        self.X = ca.MX.sym('X', self._n_x, self._N+1)
        self.U = ca.MX.sym('U', self._n_u, self._N)
        self.Z = ca.MX.sym('Z', self._n_x, self._N+1)

        self._terminal_ingredients = terminal_ingredients
        self._previous_action = [0., 0.]
        self._pose = None
        self._velocities = None
        self._target_pose = None
        self._target_velocities = None
        self.neural_states = None

        self.episode_start = True


    def set_target_velocities(self, target_velocities):
        self._target_velocities = target_velocities

    
    def set_target_pose(self, target_pose):
        self._target_pose = target_pose


    def set_current_pose(self, pose):
        self._pose = pose


    def set_current_velocities(self, velocities):
        self._velocities = velocities


    def rk4_step(self, x, u):
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + self._dt/2*k1, u)
        k3 = self._dynamics(x + self._dt/2*k2, u)
        k4 = self._dynamics(x + self._dt*k3, u)
        return x + self._dt/6*(k1 + 2*k2 + 2*k3 + k4)


    # def _compute_agent_vect(self, action):
    #     """
    #     Returns a vector describing:
    #     - distance to target (rho)
    #     - direction to target (alpha)
    #     - orientation error (delta_yaw)
    #     - error in forward speed
    #     - error in angular velocity
    #     """

    #     v_ref, gamma = action
    #     prev_v_ref, prev_gamma = self._previous_action

    #     diff_v_ref = v_ref - prev_v_ref
    #     diff_gamma = gamma - prev_gamma

    #     # Agent state
    #     pose_x, pose_y, pose_yaw = self._pose
    #     lin_vel, lat_vel, ang_vel = self._velocities

    #     # Target state
    #     target_x, target_y, target_yaw = self._target_pose
    #     target_lin_vel, target_ang_vel = self._target_velocities


    #     dx = target_x - pose_x
    #     dy = target_y - pose_y

    #     #forward error in agent space
    #     forward_error =  np.cos(pose_yaw) * dx + np.sin(pose_yaw) * dy

    #     lateral_error = -np.sin(pose_yaw) * dx + np.cos(pose_yaw) * dy
        
    #     #orientation error
    #     delta_yaw = (target_yaw - pose_yaw + np.pi) % (2 * np.pi) - np.pi

    #     target_vel_robot_frame = np.array([
    #         target_lin_vel * np.cos(target_yaw - pose_yaw),
    #         target_lin_vel * np.sin(target_yaw - pose_yaw)
    #     ])

    #     vel_error_x = target_vel_robot_frame[0] - lin_vel
    #     vel_error_y = target_vel_robot_frame[1] - lat_vel

    #     ang_vel_error = target_ang_vel - ang_vel
    
    #     agent_state_vect = np.array([
    #         forward_error,
    #         lateral_error,
    #         delta_yaw,
    #         vel_error_x,
    #         vel_error_y,
    #         ang_vel_error,
    #         gamma,
    #         v_ref
    #     ])

    #     return agent_state_vect

    def _compute_agent_vect(self, action):

        """
        Returns a vector describing:
        - distance to target (rho)
        - direction to target (alpha)
        - orientation error (delta_yaw)
        - error in forward speed
        - error in angular velocity
        """

        # Agent state
        pose_x, pose_y, pose_yaw = self._pose
        lin_vel, ang_vel = self._velocities

        # Target state
        target_x, target_y, target_yaw = self._target_pose
        target_lin_vel, target_ang_vel = self._target_velocities


        dx = target_x - pose_x
        dy = target_y - pose_y

        #forward error in agent space
        forward_error =  np.cos(pose_yaw) * dx + np.sin(pose_yaw) * dy
        # if np.abs(forward_error) > 20.0:
        #     forward_error = np.sign(forward_error) * 20.0

        #lateral error encoded as an angle
        # angle_error = np.arctan2(
        #     -np.sin(pose_yaw) * dx + np.cos(pose_yaw) * dy,
        #     np.cos(pose_yaw) * dx + np.sin(pose_yaw) * dy
        # )
        lateral_error = -np.sin(pose_yaw) * dx + np.cos(pose_yaw) * dy
        
        #orientation error
        delta_yaw = (target_yaw - pose_yaw + np.pi) % (2 * np.pi) - np.pi

        target_vel_robot_frame = np.array([
            target_lin_vel * np.cos(target_yaw - pose_yaw),
            target_lin_vel * np.sin(target_yaw - pose_yaw)
        ])

        vel_error_x = target_vel_robot_frame[0] - lin_vel

        ang_vel_error = target_ang_vel - ang_vel

        current_gamma = self._X_k['gamma']
        if np.abs(target_ang_vel) < 1e-2 or np.abs(target_lin_vel) < 1e-2: 
            target_gamma = 0.0
        else :  
            ratio = np.clip((4 * target_ang_vel) / (2 * target_lin_vel), -1.0, 1.0)
            target_gamma = 2 * np.arcsin(ratio)

        gamma_error = target_gamma - current_gamma
        gamma_error = (gamma_error + np.pi) % (2 * np.pi) - np.pi

        agent_state_vect = np.array([
            forward_error,
            lateral_error,
            delta_yaw,
            vel_error_x,
            gamma_error,
        ])

        return agent_state_vect


    def get_neural_prediction(self, obs) :

        obs = np.array(obs).reshape(1, -1)
        action, self.neural_states = self._neural_model.predict(
            obs,
            state=self.neural_states,
            episode_start=np.array([self.episode_start]),
            deterministic=True
        )
        self.episode_start = False
        return action.tolist()


    def build_nlp(self, x_init, u_ref):


        cost = 0
        g_equalities = []
        g_ineq = []


        for k in range(self._N):

            d_f = self.gamma**k
            command_error = self.U[:, k] - u_ref[:, k]
            cost += d_f * ca.mtimes([command_error.T, self._R, command_error])



        if self._terminal_ingredients is not None:

            P = self._terminal_ingredients["P"]
            state_error_terminal = self.X[:, self._N] - x_ref[:, int(self._N)]
            alpha = float(self._terminal_ingredients["alpha"])
            quad = ca.mtimes([state_error_terminal.T, ca.DM(P), state_error_terminal])
            g_ineq += [quad - alpha]

            cost += quad


        # ---------------- Constraints ----------------

        # equality constraints : dynamique

        for k in range(self._N):
            x_next_pred = self.rk4_step(self.X[:, k], self.U[:, k])
            g_equalities.append(self.X[:, k+1] - x_next_pred)

        # initial condition
        g_equalities.append(self.X[:,0] - x_init)
        g_equalities = ca.vertcat(*g_equalities)

        # inequality constraints : bornes
        u_min = self._bounds["u_min"]
        u_max = self._bounds["u_max"]
        x_min = self._bounds["x_min"]
        x_max = self._bounds["x_max"]


        for k in range(self._N):

            for i in range(self._n_u):
                g_ineq += [u_min[i] - self.U[i, k]]
                g_ineq += [self.U[i, k] - u_max[i]]


            for i in range(self._n_x):
                g_ineq += [x_min[i] - self.X[i, k+1]]
                g_ineq += [self.X[i, k+1] - x_max[i]]

        g_ineq = ca.vertcat(*g_ineq)


        opt_vars = ca.vertcat(ca.reshape(self.X, -1, 1),
                              ca.reshape(self.U, -1, 1),
                              ca.reshape(self.Z, -1, 1))

        nlp = {'x': opt_vars, 'f': cost, 'g': ca.vertcat(g_equalities, g_ineq)}
        return nlp, g_equalities, g_ineq


    def solve(self, x_init, u_ref):

        nlp, g, g_ineq = self.build_nlp(x_init, u_ref)

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp)

        lbg = [0]*g.numel() + [-ca.inf]*g_ineq.numel()
        ubg = [0]*g.numel() + [0]*g_ineq.numel()

        X0 = np.tile(x_init.reshape(-1,1), (1, self._N+1))
        U0 = np.zeros((self._n_u, self._N))
        Z0 = np.zeros((self._n_x, self._N+1))

        x_init_tile = X0.flatten(order='F')
        u_init_tile = U0.flatten(order='F')
        z_init_tile = Z0.flatten(order='F')

        initial_guess = np.concatenate([x_init_tile, u_init_tile, z_init_tile])

        sol = solver(x0=initial_guess, lbg=lbg, ubg=ubg)
        w_opt = np.array(sol['x']).squeeze()

        X_opt = w_opt[:self._n_x*(self._N+1)].reshape((self._n_x, self._N+1), order='F')
        U_opt = w_opt[self._n_x*(self._N+1):self._n_x*(self._N+1)+self._n_u*self._N].reshape((self._n_u, self._N), order='F')

        return X_opt, U_opt


    def act(self, x_target, u_target) :

        u_ref = np.zeros((2, horizon + 1))

        # step 0 : compute agent_vect from current state and target
        obs = self._compute_agent_vect(self._previous_action)
        u_init = self.get_neural_prediction(obs)[0]
        v_ref, gamma = u_init
        u_ref[:,0] = [v_ref, (gamma - self._previous_action[1]) / self._dt]

        for k in range(1, self._N):

            gamma_dot = (gamma - self._previous_action[1]) / self._dt
            u = [v_ref, gamma_dot]
            
            #simulate next state

            x_dot = self._dynamics(x_init, u)
            x_init = self.rk4_step(x_init, u)
            
            x_dot, y_dot, theta_dot, gamma_dot = x_dot
            x, y, theta, gamma = x_init
            self._pose = [x, y, theta]
            self._velocities = [x_dot, theta_dot]
            self._target_velocities =  u_target[:, k]
            self._target_pose = x_target[:, k]

            self._previous_action = [v_ref, gamma]
            obs = self._compute_agent_vect(self._previous_action)
            u_init = self.get_neural_prediction(obs)[0]
            
            v_ref, gamma = u_init
            u_ref[:, k] = [v_ref, (gamma - self._previous_action[1]) / self._dt]

        X_opt, U_opt = self.solve(x_init, u_ref)
        opt_cmd = U_opt[:, 0]

        return opt_cmd
        


if __name__ == "__main__":

    import random
    import time
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Ellipse

    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    from heracles_planning.paths.path_spiral import Path_Spiral
    from heracles_planning.paths.path_pose import Path_Pose
    from heracles_planning.paths.path_interpolation import Path_Interpolation
    from heracles_planning.trajectory import Trajectory
    from heracles_planning.navigation_planning.navigation_planner import NavigationPlanner

    from project_manager.project_manager import ProjectManager

    from heracles_forward_models.models.wheeled_loader_navigation_2025_06_05 import WheeledLoaderNavigation966_2025_06_05
    from heracles_forward_models.utils import se2

    lstm_model_path = "/home/heracles-d2/training_envs/models/best_model_bo.zip"

    project_file = "/home/heracles-d2/navigation/project_manager/data/terrain_test_gargenville_complete_reduced.hr"
    project = ProjectManager.load_from_file(project_file)


    navigation_planner = NavigationPlanner.from_project(
        project,
        resolution = 3.0, # meters
        n_cap = 8
    )


    Lf = 1.775
    Lr = 1.775
    params = {'Lf': Lf, 'Lr': Lr}
    #   Loader constraints
    gamma_max = 0.65
    gamma_dot_max = 1
    v_min = -3
    v_max = 3
    constraints = {'gamma_max': gamma_max, 'gamma_dot_max': gamma_dot_max, 'v_min': v_min, 'v_max': v_max}
    model = WheeledLoaderNavigation966_2025_06_05(params,constraints)


    def f_dyn(x, u):

        Lf = 1.775
        Lr = 1.775

        xf, yf, thetaf, gammaf = x[0], x[1], x[2], x[3]
        v, w = u[0], u[1]
        xf_dot = v * ca.cos(thetaf)
        yf_dot = v * ca.sin(thetaf)
        thetaf_dot = (v * ca.sin(gammaf)) / (Lf * ca.cos(gammaf) + Lr) + (w * Lr) / (Lf * ca.cos(gammaf) + Lr)
        gammaf_dot = w
        return ca.vertcat(xf_dot, yf_dot, thetaf_dot, gammaf_dot)


    horizon = 10
    control_frequency = 20

    big = 1e7

    bounds = {
            "u_min" : [-3.0, -0.2], # vmin, gammapmin
            "u_max" : [3.0, 0.2], # vmax, gammapmax
            "x_min" : [31.0, 545.2, -np.pi, -0.65], #xmin, ymin, thetamin, gammamin
            "x_max" : [75.6, 680.2, np.pi, 0.65]  #xmax, ymax, thetamax, gammamax
    }

    Q = np.diag([50., 50., 25.0, 1.0]) # State cost
    R = np.diag([2.0, 1.0])       # Control cost

    init_pose = np.array([51.0, 545.2, np.pi/4])
    final_pose = np.array([55.6, 599.4, np.pi/2])

    P =  np.array([[ 2.63159653e+00, -3.25449411e-11, -6.85308646e-13, 7.77634983e-14],
                [-3.25449411e-11,  2.63159653e+00,  8.90434150e-11, 3.83249971e-13],
                [-6.85308646e-13,  8.90434150e-11,  6.57832447e+00, -1.58190901e-12],
                [ 7.77634983e-14,  3.83249971e-13, -1.58190901e-12, 2.63158332e-01]])

    K =  np.array([[ 1.83129401e-23, -1.50241861e-12, -2.30867271e-13, -2.42484766e-15],
                [1.57799262e-14, -1.72072896e-13, -1.78946189e-13, -1.10481972e-14]])

    alpha = 0.25

    terminal_ingredients = {"P": P, "K": K, "alpha": alpha}

    PSF = PSF_MC_Controller(dynamics=f_dyn,
    horizon = horizon,
    control_frequency = control_frequency,
    neural_model = PPO.load(lstm_model_path),
    bounds = bounds,
    R = R)


    add_terminal_ellipse = False


    multi_dir_path = navigation_planner.generate_path(
            init_pose,
            final_pose
        )

    trajs = []

    for path in multi_dir_path.get_sections():
        vmax = random.uniform(1.5, 2.0)
        traj = Trajectory.trapezoidal_traj(
            spatial_path=path,
            init_speed=0.,
            end_speed=0.,
            max_speed=vmax,
            acceleration= vmax / 5
        )
        trajs.append(traj)

    v0, w0, x0, y0, yaw0 = trajs[0](t=0)
    x_init = np.array([x0, y0, yaw0, 0.0])


    low_level_control_params = {'zeta_gamma': 0.4874,
                                            'omega_n_gamma': 1.8471,
                                            'gamma_0': 0.,
                                            'tau_gamma': 0.25,
                                            'zeta_v': 0.8,
                                            'omega_n_v': 0.5642,
                                            'phi_zeta': 1.7083,
                                            'delta_zeta': 0.4,
                                            'phi_omega': 6.1297,
                                            'k_omega': 9.2081,
                                            'tau_v': 1.0,
                                            'lambda_decay': 0.1}

    gfr = np.zeros((3,3))
    gfr[:,0:3] = se2.vec_to_se2(np.array([x0, y0, yaw0]))
    X_k = {
        'gf': gfr,
        'gamma': 0.0,
        'chi': 0.0,
        'gamma_ref': 0.0,
        'gamma_ref_m1': 0.0,
        'vf': 0.0,
        'af': 0.0,
        'vf_ref_m1': 0.0,
        'vf_ref_m2': 0.0,
        'vf_ref_m3': 0.0,
        'vf_ref_m4': 0.0
    }

    X_poses = []
    Y_poses = []

    target_x = []
    target_y = []

    U = []

    all_x_ref = [] 
    all_u_ref = []

    dt = 1 / control_frequency

    for traj in trajs:
        t = 0.0
        traj_duration = traj.get_temporal_law().get_duration()

        while t < traj_duration:

            # ---------------- Construction de l'horizon ----------------
            x_ref = np.zeros((4, horizon + 1)) 
            u_ref = np.zeros((2, horizon + 1))

            for k in range(horizon + 1):
                t_pred = t + k * dt

                v, w, x, y, yaw = traj(t_pred)

                if t_pred > traj_duration:
                    t_pred = traj_duration
                    _, _, x, y, yaw = traj(traj_duration)
                    v, w = 0., 0.

                # Stocker les références


                x_ref[:, k] = np.array([x, y, yaw, 0.0])
                u_ref[:, k] = np.array([v, w])

                # Stocker la trajectoire cible (pour affichage)
                if k == 0:
                    target_x.append(x)
                    target_y.append(y)

            # ---------------- Calcul de la commande MPC ----------------
            start = time.time()
            u = PSF_MC_Controller.act(x_init, x_ref, u_ref)

            U.append(u)

            mpc_time = time.time() - start

            v_ref, gamma_dot = u

            # ---------------- Simulation de la commande avec la dynamique dans le MPC----------------
            #U_k = {'vf_ref': v_ref, 'gamma_dot_ref': gamma_dot}

            x_next = MPC.rk4_step(x_init, u)
            x_next = np.array(x_next)
            pose_x, pose_y, pose_yaw, gamma = x_next[0], x_next[1], x_next[2], x_next[3]

            # Récupération des poses pour affichage
            # gs = X_k['gf']
            # pose_x, pose_y, pose_yaw = se2.se2_to_vec(gs)
            # gamma = X_k["gamma"]

            X_poses.append(pose_x)
            Y_poses.append(pose_y)

            # Mise à jour de l'état initial pour le prochain pas
            x_init = x_next

            all_x_ref.append(x_ref)
            all_u_ref.append(u_ref)

            # Avancer le temps
            t += dt


    def plot_terminal_set(ax, P, alpha, center, **kwargs):
        """
        Dessine l'ellipsoïde {(x,y) | (x-c)^T P (x-c) <= alpha} projetée sur (x,y).
        center : [x_ref, y_ref]
        """
        P_xy = P[:2,:2]
        Pinv = np.linalg.inv(P_xy)

        # valeurs propres et vecteurs propres
        vals, vecs = np.linalg.eigh(Pinv)

        # demi-axes
        width, height = 2*np.sqrt(alpha*vals)

        # angle en degrés
        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))

        ell = Ellipse(xy=center, width=width, height=height,
                    angle=angle, fill=False, **kwargs)
        ax.add_patch(ell)
        return ell

    # last elements of X_poses and Y_poses
    print(len(X_poses), len(Y_poses))

    ax = project.visualize(show=False)  # ax de la carte
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('2D Trajectory Animation')
    ax.grid(True)

    # Tracer points fixes
    start_point, = ax.plot(init_pose[0], init_pose[1], 'go', label='Start')  # vert = start
    goal_point, = ax.plot(final_pose[0], final_pose[1], 'ro', label='Goal')   # rouge = goal
    trajectory_line, = ax.plot([], [], '-b', label='Trajectory')
    target_line, = ax.plot([], [], '-g', label='Target')


    # add time in animation

    terminal_patch = plot_terminal_set(
        ax, P, alpha, center=final_pose[:2],
        edgecolor='r', linestyle='--', linewidth=2, label="Terminal set"
    )


    all_x_ref = np.array(all_x_ref)
    all_u_ref = np.array(all_u_ref)

    # Sauvegarde dans un fichier .npz
    np.savez('references_traj.npz', x_ref=all_x_ref, u_ref=all_u_ref)

    ax.legend()


    # Fonction d'animation
    def animate(i):
        trajectory_line.set_data(X_poses[:i+1], Y_poses[:i+1])
        target_line.set_data(target_x[:i+1], target_y[:i+1])

        if add_terminal_ellipse:

            if i < len(target_x):
                center = [target_x[i], target_y[i]]
                terminal_patch.set_center(center)

            return trajectory_line, target_line, terminal_patch

        return trajectory_line, target_line

    # Création de l'animation
    dt = 1 / control_frequency  # secondes
    interval_ms = dt * 1000  # conversion en millisecondes pour FuncAnimation
    anim = FuncAnimation(ax.figure, animate, frames=len(X_poses), interval=interval_ms, blit=True)


    # plot command signal in an other figure
    fig2, ax2 = plt.subplots(2, 1, figsize=(8, 6))
    ax2[0].set_title('Command signals over time')
    ax2[0].set_xlabel('Time step')
    ax2[0].set_ylabel('Velocity (m/s)')
    ax2[0].grid(True)
    ax2[0].plot([u[0] for u in U], '-b', label='Velocity command (m/s)')
    ax2[0].legend() 
    ax2[1].set_xlabel('Time step')
    ax2[1].set_ylabel('Steering rate (rad/s)')
    ax2[1].grid(True)
    ax2[1].plot([u[1] for u in U], '-r', label='Steering rate command (rad/s)')
    ax2[1].legend()


    plt.show()


