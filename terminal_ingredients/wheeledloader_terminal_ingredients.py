import numpy as np
import cvxpy as cp
import casadi as ca

from scipy.optimize import minimize
from numpy.linalg import norm

# --------------------------
# utilitaires numériques
# --------------------------
def numerical_jacobian_phi(phi, x0, u0, eps=1e-6):
    """Renvoie (A,B) = d/dx phi(x0,u0), d/du phi(x0,u0) par différences finies."""
    x0 = np.atleast_1d(x0)
    u0 = np.atleast_1d(u0)
    n = x0.size
    m = u0.size
    fx = phi(x0, u0)
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    for i in range(n):
        dx = np.zeros_like(x0); dx[i] = eps
        A[:, i] = (phi(x0+dx, u0) - fx) / eps
    for j in range(m):
        du = np.zeros_like(u0); du[j] = eps
        B[:, j] = (phi(x0, u0+du) - fx) / eps
    return A, B


def f_dyn(x, u):
    Lf = 1.775
    Lr = 1.775
    xf, yf, thetaf, gammaf = x
    v, w = u
    xf_dot = v * np.cos(thetaf)
    yf_dot = v * np.sin(thetaf)
    thetaf_dot = (v * np.sin(gammaf)) / (Lf*np.cos(gammaf) + Lr) + (w * Lr) / (Lf*np.cos(gammaf) + Lr)
    gammaf_dot = w
    return np.array([xf_dot, yf_dot, thetaf_dot, gammaf_dot])


def rk4_step(x, u, dt):
    k1 = f_dyn(x, u)
    k2 = f_dyn(x + dt/2*k1, u)
    k3 = f_dyn(x + dt/2*k2, u)
    k4 = f_dyn(x + dt*k3, u)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def r_of_x(x, A, B, K):
    u = K @ x
    return f_dyn(x,u) - (A @ x + B @ u)

def R_of_x(x, A, B, K, P, kappa):
    r = r_of_x(x, A, B, K)
    return r.T@P@r + 2*x.T@(A+B@K).T@P@r - kappa*x.T@P@x


def sample_on_ellipsoid(alpha, n_samples=200):
    # échantillons sur {x : x^T P x <= alpha}
    vals, vecs = np.linalg.eigh(P)
    inv_sqrt = vecs @ np.diag(1/np.sqrt(vals)) @ vecs.T
    samples = []
    for _ in range(n_samples):
        y = np.random.randn(n_x)
        y /= np.linalg.norm(y)
        r = np.random.rand()**(1/n_x)
        y *= r
        x = np.sqrt(alpha)*(inv_sqrt @ y)
        samples.append(x)
    return samples


def compute_terminal_ingredients(Q, R, kappa=0.05, n_x=4, n_u=2, M=1, verbose=False):
    """
    Calcule les ingrédients du coût terminal (P,K,alpha) pour la MPC.
    Q : matrice de coût d'état (n_x,n_x)
    R : matrice de coût de contrôle (n_u,n_u)
    kappa : taux de décroissance
    Renvoie un dict avec les clés "P", "K", "alpha".
    """

    A, B = numerical_jacobian_phi(f_dyn, np.zeros(n_x), np.zeros(n_u))


    # 2. Variables SDP
    Oj = [cp.Variable((n_x, n_x), symmetric=True) for _ in range(M)]
    Yj = [cp.Variable((n_u, n_x)) for _ in range(M)]
    alpha_bar = cp.Variable(nonneg=True)


    zero_xx = np.zeros((n_x, n_x))
    zero_xu = np.zeros((n_x, n_u))
    zero_ux = np.zeros((n_u, n_x))
    zero_uu = np.zeros((n_u, n_u))

    Q_inv = np.linalg.inv(Q)
    R_inv = np.linalg.inv(R)


    constraints = []

    for j in range(M):
        j_next = (j + 1) % M
        AO_BY = A @ Oj[j] + B @ Yj[j]

        Mblock = cp.bmat([
            [(1 - kappa) * Oj[j],           (A @ Oj[j] + B @ Yj[j]).T,  Oj[j],           Yj[j].T],
            [A @ Oj[j] + B @ Yj[j],         Oj[j_next],                 np.zeros((n_x,n_x)), np.zeros((n_x,n_u))],
            [Oj[j],                         np.zeros((n_x,n_x)),        alpha_bar * Q_inv,   np.zeros((n_x,n_u))],
            [Yj[j],                         np.zeros((n_u,n_x)),        np.zeros((n_u,n_x)), alpha_bar * R_inv]
        ])

        constraints.append(Mblock >> 0)
        constraints.append(Oj[j] >> 1e-6 * np.eye(n_x))  # PD

    # 4. Objectif : max volume des ensembles terminaux
    obj = cp.Maximize(cp.sum([cp.log_det(Oj[j]) for j in range(M)]) - alpha_bar)

    # 5. Résolution
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=verbose)

    if verbose:
        print("Status:", prob.status)

    P_list, K_list = [], []
    for j in range(M):
        O_val = Oj[j].value
        Y_val = Yj[j].value
        Pj = np.linalg.inv(O_val)
        Kj = Y_val @ Pj
        P_list.append(Pj)
        K_list.append(Kj)

        if verbose:
            print(f"P[{j}] =\n", Pj)
            print(f"K[{j}] =\n", Kj)

    return {"P": P_list, "K": K_list, "alpha": 1/alpha_bar.value}






if __name__ == "__main__":

    Q = np.diag([10, 10, 25.0, 1.0]) # State cost
    R = np.diag([5.0, 1.0])       # Control cost

    terminal_ingredients = compute_terminal_ingredients(Q, R, kappa=0.05)

    print("P =\n", terminal_ingredients["P"])
    print("K =\n", terminal_ingredients["K"])
    print("alpha =", terminal_ingredients["alpha"])
