import optuna
import random
import time
import os 
import sys
import numpy as np
import casadi as ca

from heracles_planning.paths.path_pose import Path_Pose
from heracles_planning.paths.path_interpolation import Path_Interpolation
from heracles_planning.trajectory import Trajectory
from heracles_planning.navigation_planning.navigation_planner import NavigationPlanner

from project_manager.project_manager import ProjectManager

from heracles_forward_models.models.wheeled_loader_navigation_2025_06_05 import WheeledLoaderNavigation966_2025_06_05
from heracles_forward_models.utils import se2

project_root_path = "/".join(__file__.split("/")[:-1])
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from controller.MPC_wheeledloader import MPC_Controller


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


bounds = {
        "u_min" : [-2.0, -0.2], # vmin, gammapmin
        "u_max" : [2.0, 0.2], # vmax, gammapmax
        "x_min" : [412331.0, 5426545.2, -np.pi, -0.65], #xmin, ymin, thetamin, gammamin
        "x_max" : [412375.6, 5426595.2, np.pi, 0.65]  #xmax, ymax, thetamax, gammamax
}

init_pose = np.array([412351.0, 5426565.2, np.pi/4])
final_pose = np.array([412355.6, 5426599.4, np.pi/2])

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
                                        'gamma_0': -0.0260,
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



def objective(trial) :

    control_frequency = trial.suggest_int("coeff_control_frequency", 1, 25)
    horizon = trial.suggest_int("horizon", 5, 50)

    coeff_pose_x = trial.suggest_float("coeff_pose_x", 0.1, 50.0)
    coeff_pose_y = trial.suggest_float("coeff_pose_y", 0.1, 50.0)
    coeff_pose_theta = trial.suggest_float("coeff_pose_theta", 0.1, 50.0)
    coeff_pose_gamma = trial.suggest_float("coeff_pose_gamma", 0.1, 10.0)

    coeff_r_v = trial.suggest_float("coeff_r_v", 0.1, 50.0)
    coeff_r_w = trial.suggest_float("coeff_r_w", 0.1, 50.0)

    Q = np.diag([coeff_pose_x, coeff_pose_y, coeff_pose_theta, coeff_pose_gamma])
    R = np.diag([coeff_r_v, coeff_r_w])

    MPC = MPC_Controller(dynamics=f_dyn,
    horizon = horizon,
    control_frequency = control_frequency,
    bounds = bounds,
    Q = Q,
    R = R)

    dt = 1 / control_frequency

    X_poses = []
    Y_poses = []
    target_x = []
    target_y = []

    x_init = np.array([x0, y0, yaw0, 0.0])

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

    J = 0.0
    prev_v_cmd = 0.0
    prev_w_cmd = 0.0

    for traj in trajs:
        t = 0.0
        traj_duration = traj.get_temporal_law().get_duration()

        while t < traj_duration:

            # ---------------- Construction de l'horizon ----------------
            x_ref = np.zeros((4, horizon + 1))  # n_x = 4
            u_ref = np.zeros((2, horizon + 1))

            for k in range(horizon + 1):
                t_pred = t + k * dt
                if t_pred > traj_duration:
                    t_pred = traj_duration

                v, w, x, y, yaw = traj(t_pred)
                x_ref[:, k] = np.array([x, y, yaw, 0.0])  # gammaf = 0 si non prévu
                u_ref[:, k] = np.array([v, w])

                # Stocker la trajectoire cible (pour affichage)
                if k == 0:
                    target_x.append(x)
                    target_y.append(y)

            # ---------------- Calcul de la commande MPC ----------------
            start = time.time()
            u = MPC.act(x_init, x_ref, u_ref)
            mpc_time = time.time() - start
            #print(f"MPC solve time: {mpc_time:.3f} s")

            v_ref, gamma_dot = u

            # ---------------- Simulation du modèle ----------------
            U_k = {'vf_ref': v_ref, 'gamma_dot_ref': gamma_dot}
            W_k = {'w_a': 0.0, 'alpha': 0.0, 'beta': 0.0}

            X_k = model._simulate(X_k, U_k, W_k, low_level_control_params, 'exp')

            # Récupération des poses pour affichage
            gs = X_k['gf']
            pose_x, pose_y, pose_yaw = se2.se2_to_vec(gs)
            gamma = X_k["gamma"]
            vel = X_k["vf"]

            X_poses.append(pose_x)
            Y_poses.append(pose_y)

            # Mise à jour de l'état initial pour le prochain pas

            J += (pose_x - x_ref[0,0])**2 + (pose_y - x_ref[1,0])**2
            J += (vel - u_ref[0,0])**2
            
            # penalize variations in commands

            J += 0.1 * (v_ref - prev_v_cmd)**2 + 0.1 * (gamma_dot - prev_w_cmd)**2
            prev_v_cmd = v_ref
            prev_w_cmd = gamma_dot
            
            x_init = np.array([pose_x, pose_y, pose_yaw, gamma])

            # Avancer le temps
            t += dt

    return J


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
