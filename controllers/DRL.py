import numpy as np

class DRL_Controller:

    def __init__(self, model, control_frequency):

        self._neural_model = model
        self._control_frequency = control_frequency
        self._dt = 1 / control_frequency

        self._previous_action = [0.0, 0.0]
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
        action, self.neural_states = self._neural_model.predict(
            obs,
            state=self.neural_states,
            episode_start=np.array([self.episode_start]),
            deterministic=True
        )
        self.episode_start = False
        return action.tolist()



    def act(self, action) :

        obs = self._compute_agent_vect(action)
        u_init = self.get_neural_prediction(obs)[0]
        return u_init


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

    from heracles_forward_models.models.wheeled_loader_navigation_2025_11_05 import WheeledLoaderNavigation966_2025_11_05
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


    init_pose = np.array([51.0, 545.2, np.pi/4])
    final_pose = np.array([55.6, 599.4, np.pi/2])

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

    gs = Y['gs']
    ang_vel = Y['ang_vel']
    lat_vel = Y['lat_vel']
    lin_vel = Y['lin_vel']

    pose_x, pose_y, pose_yaw = se2.se2_to_vec(gs)
    gamma = X_k["gamma"]


    X_poses = []
    Y_poses = []

    target_x = []
    target_y = []

    prev_v, prev_gamma = 0., 0.

    control_frequency = 4
    dt = 1 / control_frequency

    DRL = DRL_Controller(PPO.load(lstm_model_path), control_frequency)

    for traj in trajs:
        t = 0.0
        traj_duration = traj.get_temporal_law().get_duration()

        while t < traj_duration:

            v, w, x, y, yaw = traj(t)

            DRL.set_current_velocities([lin_vel, ang_vel])
            DRL.set_current_pose([pose_x, pose_y, pose_yaw])
            DRL.set_target_pose([x, y, yaw])
            DRL.set_target_velocities([v, w])

            u_ref = [prev_v, prev_gamma]

            u = DRL.act(u_ref)
            v_ref, gamma_cmd = u

            U_k = {'vf_ref': v_ref, 'gamma_cmd': gamma_cmd}

            print(U_k)

            X_k = model._simulate(X_k, U_k, W_k, low_level_control_params)

            # Récupération des poses pour affichage

            Y = model.output_function(X_k, W_k)

            gs = Y['gs']
            ang_vel = Y['ang_vel']
            lin_vel = Y['lin_vel']

            pose_x, pose_y, pose_yaw = se2.se2_to_vec(gs)

            X_poses.append(pose_x)
            Y_poses.append(pose_y)
            target_x.append(x)
            target_y.append(y)

            prev_v, prev_gamma = v_ref, gamma

            # Avancer le temps
            t += dt



    ax = project.visualize(show=False)  # ax de la carte
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('2D Trajectory Animation')
    ax.grid(True)

    start_point, = ax.plot(init_pose[0], init_pose[1], 'go', label='Start')  # vert = start
    goal_point, = ax.plot(final_pose[0], final_pose[1], 'ro', label='Goal')   # rouge = goal
    trajectory_line, = ax.plot([], [], '-b', label='Trajectory')
    target_line, = ax.plot([], [], '-g', label='Target')


    add_terminal_ellipse = False

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

    plt.show()