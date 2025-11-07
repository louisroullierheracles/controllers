import numpy as np
import casadi as ca


class PID():

    def __init__(self, dt, dynamics, kp=1.0, ki=0.0, kd=0.0, windup=10, integration_leak=0.98, look_up_table=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.windup = windup
        self.integration_leak = integration_leak
        self._dynamics = dynamics
        self.integrated = 0
        self.previous_time = None
        self.previous_error = None
        self.theta = 0.95
        self._dt = dt

        self.look_up_table = look_up_table


    def apply_windup(self):
        if (self.integrated > self.windup):
            self.integrated = self.windup
        if (self.integrated < -self.windup):
            self.integrated = -self.windup


    def sign(self, value):
        if (value < 0):
            return -1
        return 1

    def rk4_step(self, x, u):
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + self._dt/2*k1, u)
        k3 = self._dynamics(x + self._dt/2*k2, u)
        k4 = self._dynamics(x + self._dt*k3, u)
        return x + self._dt/6*(k1 + 2*k2 + 2*k3 + k4)


    def __call__(self, error):

        t = time.time()

        if self.previous_time is not None:
            delta_t = t - self.previous_time
            self.integrated += delta_t * error
            self.apply_windup()
            if self.sign(error) != self.sign(self.integrated):
                self.integrated *= self.integration_leak

        self.previous_time = t

        if self.previous_error is not None:
            derivative = (error - self.previous_error) / delta_t
        else:
            derivative = 0.0

        self.previous_error = error

        command = self.kp * error + self.ki * self.integrated + self.kd * derivative

        # print(f"before look up table: {command}")
        if self.look_up_table is not None:
            corrected_command = self.look_up_table(command)
            return float(corrected_command)
            # print(f"after look up table: {corrected_command}")
        corrected_command = command

        return float(corrected_command)


if __name__ == "__main__":

    import random
    import time
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Ellipse


    from heracles_planning.paths.path_spiral import Path_Spiral
    from heracles_planning.paths.path_pose import Path_Pose
    from heracles_planning.paths.path_interpolation import Path_Interpolation
    from heracles_planning.trajectory import Trajectory
    from heracles_planning.navigation_planning.navigation_planner import NavigationPlanner

    from project_manager.project_manager import ProjectManager

    from heracles_forward_models.models.wheeled_loader_navigation_2025_06_05 import WheeledLoaderNavigation966_2025_06_05
    from heracles_forward_models.utils import se2




    project_file = "/home/heracles-d2/navigation/project_manager/data/terrain_test_gargenville_complete.hr"
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

    init_pose = np.array([412351.0, 5426565.2, np.pi/4])
    final_pose = np.array([412355.6, 5426599.4, np.pi/2])

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
    print("x_init:", x_init)


    X_poses = []
    Y_poses = []

    target_x = []
    target_y = []

    control_frequency = 10  # Hz

    dt = 1 / control_frequency

    controller = PID(dt, f_dyn, kp=2.0, ki=0.1, kd=0.05, windup=20, integration_leak=0.98)

    for traj in trajs:
        t = 0.0
        traj_duration = traj.get_temporal_law().get_duration()

        while t < traj_duration:

            v_ref, w_ref, x_ref, y_ref, yaw_ref = traj(t)

            v = controller(np.sqrt((x_ref - x_init[0])**2 + (y_ref - x_init[1])**2))
            gamma_dot = controller(yaw_ref - x_init[2])

            # ---------------- Simulation de la commande avec la dynamique dans le MPC----------------
            #U_k = {'vf_ref': v_ref, 'gamma_dot_ref': gamma_dot}

            u = np.array([v, gamma_dot])

            x_next = controller.rk4_step(x_init, u)
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


    # terminal_patch = plot_terminal_set(
    #     ax, P, alpha, center=final_pose[:2],
    #     edgecolor='r', linestyle='--', linewidth=2, label="Terminal set"
    # )


    ax.legend()


    # Fonction d'animation
    def animate(i):
        trajectory_line.set_data(X_poses[:i+1], Y_poses[:i+1])
        target_line.set_data(target_x[:i+1], target_y[:i+1])

        return trajectory_line, target_line

    # Création de l'animation
    dt = 1 / control_frequency  # secondes
    interval_ms = dt * 1000  # conversion en millisecondes pour FuncAnimation
    anim = FuncAnimation(ax.figure, animate, frames=len(X_poses), interval=interval_ms, blit=True)

    plt.show()
