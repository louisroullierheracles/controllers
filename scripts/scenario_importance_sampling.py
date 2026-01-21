import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import casadi as ca
import os
import sys
import csv
import json

from heracles_planning.paths.path_spiral import Path_Spiral
from heracles_planning.paths.path_pose import Path_Pose
from heracles_planning.paths.path_b_spline import Path_B_Spline
from heracles_planning.paths.path_spline import Path_Spline
from heracles_planning.paths.path_interpolation import Path_Interpolation
from heracles_planning.trajectory import Trajectory


project_root_path = "/".join(__file__.split("/")[:-1])
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from controllers.MPC import MPC_Controller
from controllers.PSF_MC_real_model import PSF_MC_Controller
from controllers.PSF import PSF_Controller
#from controllers.EPSF import EPSF_Controller
#from controllers.EPSF_stable import EPSF_Controller
from controllers.EPSF_obstacles_new_formulation import EPSF_Controller
from controllers.DRL import DRL_Controller
from controllers.backup_policy import Backup_Policy

from project_manager.project_manager import ProjectManager
from heracles_forward_models.models.wheeled_loader_navigation_2025_11_05 import WheeledLoaderNavigation966_2025_11_05
from heracles_forward_models.utils import se2


points = [
    np.array([-0.0, 0.0]),
    np.array([10.0, 0.0]),
    np.array([20., 0.0]),
]


vmax = 1.2

path_2d = Path_Spline(points)

#path_2d.plot_with_curvature(show=True)

path = Path_Pose(path_2d = path_2d, is_forward=True)
traj = Trajectory.trapezoidal_traj(
    spatial_path=path,
    init_speed=0.,
    end_speed=0.,
    max_speed=vmax,
    acceleration= vmax / 5.0
)

def g_zheta(v) :

    return 1 + 1.7083 * np.exp(-(v / 0.4) ** 2)

def g_omega(v, v_ref) :

    return 1 + 6.1297 / (1 + np.exp(-9.2081 * (np.abs(v) - np.abs(v_ref))))

def g_zheta_ca(v) :

    return 1 + 1.7083 * ca.exp(-(v / 0.4) ** 2)

def g_omega_ca(v, v_ref) :
    
    return 1 + 6.1297 / (1 + ca.exp(-9.2081 * (ca.fabs(v) - ca.fabs(v_ref))))


def f_dyn_simu(x, u) :

    Lf = 1.775
    Lr = 1.775

    vf, chi, xf, yf, thetaf, gamma, af = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    v_d, gamma_d = u[0], u[1]

    af_dot = - 2 * 0.800 * 0.5642 * g_zheta(vf) * af - (0.5642 ** 2) * g_omega(vf, v_d) * (vf - v_d)
    vf_dot = af
    xf_dot = vf * np.cos(thetaf)
    yf_dot = vf * np.sin(thetaf)
    chi_dot = -2 * 0.4874 * 1.841 * chi - (1.841**2) * gamma + (1.841**2) * gamma_d 
    gamma_dot = chi

    thetaf_dot = (vf * np.sin(gamma) + Lr * chi) / (Lf * np.cos(gamma) + Lr)

    return np.array([vf_dot, chi_dot, xf_dot, yf_dot, thetaf_dot, gamma_dot, af_dot])


def f_dyn(x, u) :
    
    Lf = 1.775
    Lr = 1.775

    vf, chi, xf, yf, thetaf, gamma, af = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    v_d, gamma_d = u[0], u[1]

    af_dot = - 2 * 0.800 * 0.5642 * g_zheta_ca(vf) * af - (0.5642 ** 2) * g_omega_ca(vf, v_d) * (vf - v_d)
    vf_dot = af
    xf_dot = vf * ca.cos(thetaf)
    yf_dot = vf * ca.sin(thetaf)
    chi_dot = -2 * 0.4874 * 1.841 * chi - (1.841**2) * gamma + (1.841**2) * gamma_d 
    gamma_dot = chi

    thetaf_dot = (vf * ca.sin(gamma) + Lr * chi) / (Lf * ca.cos(gamma) + Lr)

    return ca.vertcat(vf_dot, chi_dot, xf_dot, yf_dot, thetaf_dot, gamma_dot, af_dot)


def rk4_step(model, x, u):
    dt = 1 / control_frequency
    k1 = model(x, u)
    k2 = model(x + (dt/2)*k1, u)
    k3 = model(x + (dt/2)*k2, u)
    k4 = model(x + dt*k3, u)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


bounds = {
        "u_min" : [-2.5, -0.65], # vmin, gammapmin
        "u_max" : [2.5, 0.65], # vmax, gammapmax
        "x_min" : [-2.0, -1.5, -50.0, -50.0, -3.141, -0.65, -1.],
        "x_max" : [2.0, 1.5, 50.0, 50.0, 3.141, 0.65, 1.]
}

obstacles = [
        # {"x_min" : 5.0, 
        # "x_max" : 6.5,
        # "y_min" : -0.5,
        # "y_max" : 10},
        # {"x_min" : 9.5, 
        # "x_max" : 15.5,
        # "y_min" : 3.5,
        # "y_max" : 5.0},
        # {"x_min" : 10.5, 
        # "x_max" : 12.0,
        # "y_min" : -10.0,
        # "y_max" : 0.5},
        # {
        # "x_min" : 15.0,
        # "x_max" : 17.0,
        # "y_min" : -10.0,
        # "y_max" : 0.
        # }
]

def dist_to_obstacle(point, obstacle):
    x, y = point
    x_min = obstacle['x_min']
    x_max = obstacle['x_max']
    y_min = obstacle['y_min']
    y_max = obstacle['y_max']

    if x < x_min:
        dx = x_min - x
    elif x > x_max:
        dx = x - x_max
    else:
        dx = 0

    if y < y_min:
        dy = y_min - y
    elif y > y_max:
        dy = y - y_max
    else:
        dy = 0

    return np.sqrt(dx**2 + dy**2)


horizon = 20
# random R
R = np.diag([3., 1.])
Q = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 2., 0.])
Q_t = np.diag([.0, 0.0, 1., 1.0, 1.0])

control_frequency = 2.0

EPSF = EPSF_Controller(
    dynamics=f_dyn,
    horizon=horizon,
    control_frequency=control_frequency,
    neural_model_path="/home/heracles-d2/training_envs/models/ppo_analytic_model_N_20_2025_12_26.zip",
    bounds=bounds,
    obstacles=obstacles,
    backup_policy=None,
    R=R,
    diff_R = np.diag([1.0, 1.0]),
    Q = Q,
    Q_t = Q_t,
)


v0, w0, x0, y0, yaw0 = traj(t=0)
x_init = np.array([v0, w0, x0, y0, yaw0, 0.0, 0.0])
lin_vel = v0
ang_vel = w0
pose_x = x0
pose_y = y0
pose_yaw = yaw0
dv = v0
dgamma = 0.0

init_pose = [x0, y0, yaw0]
final_pose = traj(t=traj.get_temporal_law().get_duration())[2:5]

X_poses = []
Y_poses = []
X_poses_drl = []
Y_poses_drl = []
target_x = []
target_y = []
curvature_radius_agent = []

dt = 1 / control_frequency

t = 0.0
i = 0
traj_duration = traj.get_temporal_law().get_duration()

folder = "data/importance_sampling"
if not os.path.exists(folder):
    os.makedirs(folder)


params = {
    "horizon": horizon,
    "control_frequency": control_frequency,
    "R": R.tolist(),
    "Q": Q.tolist(),
    "date" : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "diff_commands_penalization": False,
    "PSF_type": "Extended PSF"
}

json_title = "data/importance_sampling/experiment_params.json"
# add the date to the json title
json_title = json_title.replace(".json", f"_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.json")

with open(json_title, "w") as json_file:
    json.dump(params, json_file, indent=4)

csv_title = "data/importance_sampling/classic_psf.csv"
# add the date to the csv title
csv_title = csv_title.replace(".csv", f"_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.csv")

target_vector = np.zeros((5, horizon))
first_time = True

k = 0

with open(csv_title, "w", newline="") as f:

    writer = csv.writer(f)
    header = ["step", "x", "y", "theta", "v", "gamma", "chi", "accel", "target_x", "target_y", "target_yaw"]
    for i in range(horizon):
        header.append(f"u_rl_v_{i}")
        header.append(f"u_rl_gamma_{i}")
        header.append(f"x_opt_{i}")
        header.append(f"y_opt_{i}")
        header.append(f"theta_opt_{i}")
        header.append(f"v_opt_{i}")
        header.append(f"gamma_opt_{i}")
        header.append(f"chi_opt_{i}")
        header.append(f"accel_opt_{i}")
    header.append("black_state")

    writer.writerow(header)


    while t < traj_duration :

        print(t)

        v, w, x, y, yaw = traj(t)

        EPSF.set_current_velocities([x_init[0],  x_init[1]])
        EPSF.set_current_pose([pose_x, pose_y, pose_yaw])
        EPSF.set_gamma(dgamma)

        for i in range(horizon):
            t_future = t + (i * dt)
            v_ref, w_ref, x_ref, y_ref, yaw_ref = traj(t_future)
            yaw_ref = (yaw_ref + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
            target_vector[:, i] = np.array([v_ref, w_ref, x_ref, y_ref, yaw_ref])


        EPSF.set_target_vector(target_vector)

        if t == 0.0 :
            EPSF.initialize_Jb(x_init)

        # if first_time and (dist_to_obstacle((target_vector[2, -1], target_vector[3, -1]), obstacles[0]) < 2.0) :
        #     EPSF.initialize_Jb(x_init)
            #first_time = False

        u, X_opt, u_rl, U_opt_list, cost, need_init = EPSF.act(x_init)
        #u = EPSF.act_mpc(x_init)
        print("u", u)
        print("u_rl", [u_rl[0], u_rl[1]])
        
        x_next = rk4_step(f_dyn_simu, x_init, u)
        print("x_next", x_next)

        v, chi, pose_x, pose_y, pose_yaw = x_next[0], x_next[1], x_next[2], x_next[3], x_next[4]
        pose_yaw = (pose_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        X_poses.append(pose_x)
        Y_poses.append(pose_y)
        target_x.append(x)
        target_y.append(y)

        EPSF.update_Jb_from_solution(x_init, X_opt, cost, u)

        row = [k, pose_x, pose_y, pose_yaw, x_next[3], x_next[4], x_next[5], x_next[6], x, y, yaw]
        for i in range(horizon):
            row.append(U_opt_list[2*i])
            row.append(U_opt_list[2*i+1])
            row.append(X_opt[0][i])
            row.append(X_opt[1][i])
            row.append(X_opt[2][i])
            row.append(X_opt[3][i])
            row.append(X_opt[4][i])
            row.append(X_opt[5][i])
            row.append(X_opt[6][i])

        writer.writerow(row)

        x_init = x_next
        t += dt
        k += 1


ax = plt.gca()
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('2D Trajectory Animation')
ax.grid(True)

start_point, = ax.plot(init_pose[0], init_pose[1], 'go', label='Start')  # vert = start
goal_point, = ax.plot(final_pose[0], final_pose[1], 'ro', label='Goal')   # rouge = goal
trajectory_line, = ax.plot([], [], '-b', label='Trajectory')
target_line, = ax.plot([], [], '-g', label='Target')

# Boucle sur chaque obstacle et ajout à l'axe
for obs in obstacles:
    width = obs["x_max"] - obs["x_min"]
    height = obs["y_max"] - obs["y_min"]
    rect = patches.Rectangle((obs["x_min"], obs["y_min"]), width, height, 
                             linewidth=1, edgecolor='r', facecolor='blue', alpha=0.5)
    ax.add_patch(rect)

# Définition des limites de l'affichage

def animate(i):

    trajectory_line.set_data(X_poses[:i+1], Y_poses[:i+1])
    target_line.set_data(target_x[:i+1], target_y[:i+1])
    return trajectory_line, target_line

dt = 1 / control_frequency  # secondes
interval_ms = dt * 1000  # conversion en millisecondes pour FuncAnimation
anim = FuncAnimation(ax.figure, animate, frames=len(X_poses), interval=interval_ms, blit=True)



ax.set_xlim(-5, 25)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')  # Pour que les rectangles ne soient pas déformés

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Obstacles")
plt.grid(True)
plt.show()




