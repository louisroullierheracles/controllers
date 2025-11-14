import numpy as np
import casadi as ca
from scipy.spatial import ConvexHull


def convex_hull_distance(poly, x):
    """
    poly : (N,2) polygone convexe en CCW
    x    : MX(2)
    return: distance signée d(x, poly)  = max_i d_i(x)
    """
    N = poly.shape[0]
    d = -1e9  # -inf
    
    for i in range(N):
        p_i = poly[i]
        p_j = poly[(i+1) % N]

        e = p_j - p_i
        n = np.array([e[1], -e[0]])  # outward normal

        d_i = ca.dot(x - p_i, n) / np.linalg.norm(n)
        d = ca.fmax(d, d_i)

    return d


class PSF_Controller:

    def __init__(self,
                dynamics,
                horizon : int,
                control_frequency : float,
                neural_model,
                bounds : dict,
                obstacles = None,
                backup_policy = None,
                R = np.diag([1.0, 0.1]),
                terminal_ingredients=None):

        self._N = horizon
        self._dt = 1 / control_frequency

        self._bounds = bounds

        self._n_x = 3
        self._n_u = 2
        self._R = R

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



    def _compute_agent_vect(self):

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

        current_gamma = 0.
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
        action, _ = self._neural_model.predict(
            obs,
            deterministic=True
        )
        self.episode_start = False
        return action.tolist()



    def obstacle_to_ellipse(self, points: np.ndarray, safety_radius: float = 0.2):

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



    def build_nlp(self, x_init, u_ref, h):


        nx, nu = self._n_x, self._n_u
        X = ca.MX.sym('X', nx, h+1)
        U = ca.MX.sym('U', nu, h)

        cost = 0

        # diff_rl = U[:,0] - ca.DM(u_ref)
        # cost += ca.mtimes([diff_rl.T, ca.DM(self._R), diff_rl])

        #penalize variations in control
        for k in range(h):

            diff_rl = U[:,k] - ca.DM(u_ref)
            cost += ca.mtimes([diff_rl.T, ca.DM(self._R), diff_rl])

            # diff = U[:,k] - U[:,k-1]
            # cost += ca.mtimes([diff.T, ca.DM(self._R), diff])

        if self._terminal_ingredients is not None:

            P = self._terminal_ingredients["P"]
            state_error_terminal = X[:, h] - x_ref[:, h]
            alpha = float(self._terminal_ingredients["alpha"])
            quad = ca.mtimes([state_error_terminal.T, ca.DM(P), state_error_terminal])
            g_ineq += [quad - alpha]

            cost += quad


        # ---------------- Constraints ----------------

        # equality constraints : dynamique
        g_equalities = []

        for k in range(h):
            x_next_pred = self.rk4_step(X[:, k], U[:, k])
            # theta should be wrapped between -pi and pi
            x_next_pred[2] = ca.fmod(x_next_pred[2] + ca.pi, 2 * ca.pi) - ca.pi
            g_equalities.append(X[:, k+1] - x_next_pred)

        # initial condition
        g_equalities.append(X[:,0] - x_init)
        g_equalities = ca.vertcat(*g_equalities)

        g_ineq = []

        # inequality constraints : bornes
        u_min = self._bounds["u_min"]
        u_max = self._bounds["u_max"]
        x_min = self._bounds["x_min"]
        x_max = self._bounds["x_max"]


        for k in range(h):

            for i in range(self._n_u):
                g_ineq += [u_min[i] - U[i, k]]
                g_ineq += [U[i, k] - u_max[i]]


            for i in range(self._n_x):
                g_ineq += [x_min[i] - X[i, k+1]]
                g_ineq += [X[i, k+1] - x_max[i]]


        for obstacle in self.obstacles:
            ellipse = self.obstacle_to_ellipse(obstacle, safety_radius=1.0)
            center = ellipse['center']
            radius = ellipse['radius']

            for k in range(h+1):
                dist_sq = (X[0, k] - center[0])**2 + (X[1, k] - center[1])**2
                g_ineq += [radius**2 - dist_sq]

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



    def solve_nlp(self, nlp, sizes) : 

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

        X0 = np.tile(x_init.reshape(-1,1), (1, horizon+1))
        U0 = np.zeros((self._n_u, horizon))
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


    def solve_global_problem(self, x_init, u_rl) :

        x_init = np.array(x_init).reshape(-1)
        u_rl = np.array(u_rl).reshape(-1)

        feasible = False
        X_opt = None
        U_opt = None

        # unascendant horizon search

        for h in range(self._N, 0, -1):
            nlp, sizes, g_equalities, g_ineq = self.build_nlp(x_init, u_rl, h)
            X_opt, U_opt = self.solve_nlp(nlp, sizes)

            if X_opt is not None and U_opt is not None:
                return X_opt, U_opt

        X_opt = x_init.reshape(-1, 1) 
        if self.backup_policy is not None :
            U_opt = self.backup_policy.act(x_init)
            U_opt = U_opt.reshape(-1, 1)
        else :
            U_opt = np.zeros((self._n_u, 1))

        return X_opt, U_opt



    def act(self, x_init) :

        obs = self._compute_agent_vect()
        u_init = self.get_neural_prediction(obs)[0]
        print("Neural action:", u_init)

        v_ref, gamma = u_init
        u_init = np.array([v_ref, gamma])

        X_opt, U_opt = self.solve_global_problem(x_init, u_init)
        opt_cmd = U_opt[:, 0]

        return opt_cmd, u_init, X_opt



if __name__ == "__main__":


    import random
    import time
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Ellipse

    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    from backup_policy import Backup_Policy

    from heracles_planning.paths.path_spiral import Path_Spiral
    from heracles_planning.paths.path_pose import Path_Pose
    from heracles_planning.paths.path_interpolation import Path_Interpolation
    from heracles_planning.trajectory import Trajectory
    from heracles_planning.navigation_planning.navigation_planner import NavigationPlanner

    from project_manager.project_manager import ProjectManager

    from heracles_forward_models.models.wheeled_loader_navigation_2025_11_05 import WheeledLoaderNavigation966_2025_11_05
    from heracles_forward_models.utils import se2

    lstm_model_path = "/home/heracles-d2/training_envs/models/best_model_bo.zip"


    project_file = "/home/heracles-d2/navigation/project_manager/data/terrain_test_gargenville_complete_reduced.hr"
    project = ProjectManager.load_from_file(project_file)


    navigation_planner = NavigationPlanner.from_project(
        project,
        resolution = 2.0, # meters
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
    model = WheeledLoaderNavigation966_2025_11_05(params,constraints)


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


    def f_dyn(x, u):
        Lf = 1.775
        Lr = 1.775

        xf, yf, thetaf = x[0], x[1], x[2]
        v, gamma = u[0], u[1]

        xf_dot = v * ca.cos(thetaf)
        yf_dot = v * ca.sin(thetaf)
        thetaf_dot = v * ca.tan(gamma) / (Lf + Lr)

        return ca.vertcat(xf_dot, yf_dot, thetaf_dot)

    bounds = {
            "u_min" : [-3.0, -0.65], # vmin, gammapmin
            "u_max" : [3.0, 0.65], # vmax, gammapmax
            "x_min" : [31.0, 545.2, -np.pi], #xmin, ymin, thetamin, gammamin
            "x_max" : [75.6, 680.2, np.pi]  #xmax, ymax, thetamax, gammamax
    }

    R = np.diag([1.0, 1.0])  # Control cost


    obstacles = [
            np.array([
                [60., 570.000],
                [60., 565.0],
                [55., 565.0],
                [55., 570.0],
            ])
    ]


    init_pose = np.array([51.0, 545.2, np.pi/4])
    final_pose = np.array([55.6, 599.4, np.pi/2])

    multi_dir_path = navigation_planner.generate_path(
            init_pose,
            final_pose
        )

    horizon = 15
    control_frequency = 4.0


    PSF = PSF_Controller(
        dynamics=f_dyn,
        horizon=horizon,
        control_frequency=control_frequency,
        neural_model=PPO.load(lstm_model_path),
        bounds=bounds,
        obstacles=obstacles,
        backup_policy=None,
        R=R
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
    x_init = np.array([x0, y0, yaw0])

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

    W_k = {'w_a': 0.0, 'alpha': 0.0, 'beta': 0.0}


    Y = model.output_function(X_k, W_k)

    gs = X_k['gf']
    ang_vel = Y['ang_vel']
    lin_vel = Y['lin_vel']

    pose_x, pose_y, pose_yaw = se2.se2_to_vec(gs)
    gamma = X_k["gamma"]


    X_poses = []
    Y_poses = []

    target_x = []
    target_y = []
    u_psf = []
    u_drl = []

    black_states_index = []
    dt = 1 / control_frequency

    with open("psf_log.csv", "w", newline="") as f:

        writer = csv.writer(f)
        header = ["step", "x", "y", "theta", "v_ref_drl", "gamma_cmd_drl"]
        for i in range(horizon):
            #write each element of X_opt "
            header += [f"x_opt_{i}", f"y_opt_{i}", f"theta_opt_{i}"]
        header += ["black_state"]
        writer.writerow(header)

        for traj in trajs:
            
            t = 0.0
            i = 0
            traj_duration = traj.get_temporal_law().get_duration()

            while t < traj_duration:

                v, w, x, y, yaw = traj(t)

                PSF.set_current_velocities([lin_vel,  ang_vel])
                PSF.set_current_pose([pose_x, pose_y, pose_yaw])
                PSF.set_target_pose([x, y, yaw])
                PSF.set_target_velocities([v, w])

                x_init = np.array([pose_x, pose_y, pose_yaw])

                u, u_init, X_opt = PSF.act(x_init)
                v_ref, gamma_cmd = u

                u_psf.append([v_ref, gamma_cmd])
                u_drl.append(u_init)
                if not np.allclose(u, u_init, atol=1e-2):
                    black_states_index.append(i)

                U_k = {'vf_ref': v_ref, 'gamma_cmd': gamma_cmd}

                X_k = model._simulate(X_k, U_k, W_k, low_level_control_params)

                # Récupération des poses pour affichage

                Y = model.output_function(X_k, W_k)

                gs = X_k['gf']
                ang_vel = Y['ang_vel']
                lin_vel = Y['lin_vel']

                pose_x, pose_y, pose_yaw = se2.se2_to_vec(gs)
                gamma = X_k["gamma"]

                X_poses.append(pose_x)
                Y_poses.append(pose_y)
                target_x.append(x)
                target_y.append(y)

                x_next = np.array([pose_x, pose_y, pose_yaw])


                # Mise à jour de l'état initial pour le prochain pas
                x_init = x_next

                u_ref = [v_ref, gamma_cmd]
                row = [i, pose_x, pose_y, pose_yaw, u_init[0], u_init[1]]

                for k in range(horizon):
                    if X_opt is not None:
                        row += [X_opt[0, k], X_opt[1, k], X_opt[2, k]] 
                    else:
                        row += [0.0, 0.0, 0.0]
                row += [1 if i in black_states_index else 0]

                writer.writerow(row)

                # Avancer le temps
                t += dt
                i += 1


    # compare u_psf and u_drl
    u_psf = np.array(u_psf)
    u_drl = np.array(u_drl)
    time_steps = np.arange(u_psf.shape[0]) * dt
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time_steps, u_psf[:,0], label='PSF v')
    plt.plot(time_steps, u_drl[:,0], label='DRL v', linestyle='--')
    plt.ylabel('Linear Velocity (m/s)')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(time_steps, u_psf[:,1], label='PSF gamma')
    plt.plot(time_steps, u_drl[:,1], label='DRL gamma', linestyle='--')
    plt.ylabel('Steering Angle (rad)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.suptitle('Control Inputs Comparison')
    plt.show()



    ax = project.visualize(show=False)  # ax de la carte
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('2D Trajectory Animation')
    ax.grid(True)

    start_point, = ax.plot(init_pose[0], init_pose[1], 'go', label='Start')  # vert = start
    goal_point, = ax.plot(final_pose[0], final_pose[1], 'ro', label='Goal')   # rouge = goal
    trajectory_line, = ax.plot([], [], '-b', label='Trajectory')
    target_line, = ax.plot([], [], '-g', label='Target')


    ellipse = PSF.obstacle_to_ellipse(obstacles[0], safety_radius=0.2)

    # Dessiner les obstacles
    for obs in obstacles:
        polygon = plt.Polygon(obs, color='gray')
        ax.add_patch(polygon)

    # plot ellipse
    if ellipse is not None:
        center = ellipse['center']
        radius = ellipse['radius']

        terminal_patch = Ellipse(
            xy=center,
            width=2*radius,
            height=2*radius,
            angle=0,
            edgecolor='r',
            facecolor='none',
            linestyle='--',
            label='TObstacle ellipse'
        )
        ax.add_patch(terminal_patch)

    ax.relim()
    ax.autoscale_view()

    # Fixer les limites après avoir tout dessiné pour récupérer les bonnes bornes
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Créer le cercle en haut à gauche (après avoir les limites correctes)
    psf_indicator_radius = 1
    psf_indicator = plt.Circle(
        (x_min , y_max - psf_indicator_radius*2),
        psf_indicator_radius,
        color='black',
        visible=False
    )
    ax.add_patch(psf_indicator)
    ax.legend()

    # === Fonction d’animation ===
    def animate(i):
        trajectory_line.set_data(X_poses[:i+1], Y_poses[:i+1])
        target_line.set_data(target_x[:i+1], target_y[:i+1])

        # Activation du cercle noir quand le PSF est actif
        psf_indicator.set_visible(i in black_states_index)

        # Important : inclure le cercle dans les objets retournés
        return trajectory_line, target_line, psf_indicator

    # Création de l'animation
    dt = 1 / control_frequency  # secondes
    interval_ms = dt * 1000  # conversion en millisecondes pour FuncAnimation
    anim = FuncAnimation(ax.figure, animate, frames=len(X_poses), interval=interval_ms, blit=True)

    plt.show()