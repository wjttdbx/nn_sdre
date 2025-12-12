import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

MU_EARTH = 398600.4418  # km^3/s^2
RE_EARTH = 6378.137     # km


def plot_earth_sphere(ax, radius=RE_EARTH, n_u=36, n_v=18, alpha=0.25):
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="lightgray", linewidth=0, alpha=alpha)


def save_eci_gif(chief_r_eci, deputy_r_eci, out_path="eci_animation_elliptic.gif", stride=5, fps=20):
    """保存ECI三维轨迹动图（GIF）。"""
    idx = np.arange(0, chief_r_eci.shape[1], stride, dtype=int)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    plot_earth_sphere(ax)
    ax.set_xlabel("ECI x [km]")
    ax.set_ylabel("ECI y [km]")
    ax.set_zlabel("ECI z [km]")
    ax.set_title("Inertial Trajectories (ECI) - Animation")

    (chief_line,) = ax.plot([], [], [], "r", label="Target/Chief (ECI)")
    (deputy_line,) = ax.plot([], [], [], "b", label="Pursuer/Deputy (ECI)")
    chief_pt = ax.scatter([], [], [], c="r", s=20)
    deputy_pt = ax.scatter([], [], [], c="b", s=20)
    ax.legend()

    all_pts = np.hstack((chief_r_eci, deputy_r_eci))
    max_range = np.max(np.ptp(all_pts, axis=1))
    center = np.mean(all_pts, axis=1)
    half = 0.5 * max_range
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    def init():
        chief_line.set_data([], [])
        chief_line.set_3d_properties([])
        deputy_line.set_data([], [])
        deputy_line.set_3d_properties([])
        return (chief_line, deputy_line, chief_pt, deputy_pt)

    def update(frame_i):
        k = idx[frame_i]
        chief_line.set_data(chief_r_eci[0, idx[: frame_i + 1]], chief_r_eci[1, idx[: frame_i + 1]])
        chief_line.set_3d_properties(chief_r_eci[2, idx[: frame_i + 1]])
        deputy_line.set_data(deputy_r_eci[0, idx[: frame_i + 1]], deputy_r_eci[1, idx[: frame_i + 1]])
        deputy_line.set_3d_properties(deputy_r_eci[2, idx[: frame_i + 1]])

        chief_pt._offsets3d = (
            np.array([chief_r_eci[0, k]]),
            np.array([chief_r_eci[1, k]]),
            np.array([chief_r_eci[2, k]]),
        )
        deputy_pt._offsets3d = (
            np.array([deputy_r_eci[0, k]]),
            np.array([deputy_r_eci[1, k]]),
            np.array([deputy_r_eci[2, k]]),
        )
        return (chief_line, deputy_line, chief_pt, deputy_pt)

    ani = animation.FuncAnimation(fig, update, frames=idx.size, init_func=init, interval=1000 / fps, blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def two_body_accel(r_eci: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    r = norm(r_eci)
    return -mu * r_eci / (r**3)


def chief_perigee_state(a: float, e: float, mu: float = MU_EARTH) -> tuple[np.ndarray, np.ndarray]:
    """简单起始：近地点、赤道平面、速度沿+y。"""
    rp = a * (1.0 - e)
    vp = np.sqrt(mu * (1.0 + e) / (a * (1.0 - e)))
    r0 = np.array([rp, 0.0, 0.0], dtype=float)
    v0 = np.array([0.0, vp, 0.0], dtype=float)
    return r0, v0


def lvlh_dcm(chief_r_eci: np.ndarray, chief_v_eci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (C_LI, C_IL)

    C_LI: LVLH -> Inertial (列向量为 LVLH 三轴在惯性系的表达)
    C_IL: Inertial -> LVLH

    LVLH 轴定义：
    - x: 径向 ir
    - z: 角动量方向 ih
    - y: 沿轨 it = ih × ir
    """
    ir = chief_r_eci / norm(chief_r_eci)
    h = np.cross(chief_r_eci, chief_v_eci)
    ih = h / norm(h)
    it = np.cross(ih, ir)
    it = it / norm(it)
    C_LI = np.column_stack((ir, it, ih))
    C_IL = C_LI.T
    return C_LI, C_IL


def lvlh_omega(chief_r_eci: np.ndarray, chief_v_eci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LVLH 相对惯性角速度 omega 与其导数 omega_dot（均在惯性系表达）。

    对两体轨道：h 常量，omega = h / r^2，omega_dot = -2 (r_dot/r) omega
    """
    r = norm(chief_r_eci)
    h = np.cross(chief_r_eci, chief_v_eci)
    omega = h / (r**2)
    rdot = float(np.dot(chief_r_eci, chief_v_eci) / r)
    omega_dot = -2.0 * (rdot / r) * omega
    return omega, omega_dot


def inertial_from_rel(chief_r: np.ndarray, chief_v: np.ndarray, rho_l: np.ndarray, rhodot_l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C_LI, C_IL = lvlh_dcm(chief_r, chief_v)
    omega_i, _ = lvlh_omega(chief_r, chief_v)
    omega_l = C_IL @ omega_i

    deputy_r = chief_r + C_LI @ rho_l
    v_rel_i = C_LI @ (rhodot_l + np.cross(omega_l, rho_l))
    deputy_v = chief_v + v_rel_i
    return deputy_r, deputy_v


def rel_from_inertial(chief_r: np.ndarray, chief_v: np.ndarray, deputy_r: np.ndarray, deputy_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从惯性状态得到 LVLH 相对状态与旋转项。

    返回：rho, rho_dot, C_LI, omega_l, omega_dot_l
    """
    C_LI, C_IL = lvlh_dcm(chief_r, chief_v)
    omega_i, omega_dot_i = lvlh_omega(chief_r, chief_v)
    omega_l = C_IL @ omega_i
    omega_dot_l = C_IL @ omega_dot_i

    r_rel_i = deputy_r - chief_r
    v_rel_i = deputy_v - chief_v

    rho = C_IL @ r_rel_i
    rho_dot = C_IL @ v_rel_i - np.cross(omega_l, rho)

    return rho, rho_dot, C_LI, omega_l, omega_dot_l


def rel_xdot_uncontrolled(chief_r: np.ndarray, chief_v: np.ndarray, x_rel: np.ndarray) -> np.ndarray:
    """固定 chief_r/chief_v 时，计算 LVLH 相对动力学 xdot（不含控制）。

    x_rel = [rho(3), rho_dot(3)]
    """
    rho = x_rel[0:3]
    rho_dot = x_rel[3:6]

    deputy_r, deputy_v = inertial_from_rel(chief_r, chief_v, rho, rho_dot)

    a_c = two_body_accel(chief_r)
    a_d = two_body_accel(deputy_r)
    a_rel_i = a_d - a_c

    _, _, C_LI, omega_l, omega_dot_l = rel_from_inertial(chief_r, chief_v, deputy_r, deputy_v)
    C_IL = C_LI.T

    rho_ddot = (
        C_IL @ a_rel_i
        - 2.0 * np.cross(omega_l, rho_dot)
        - np.cross(omega_dot_l, rho)
        - np.cross(omega_l, np.cross(omega_l, rho))
    )

    return np.hstack((rho_dot, rho_ddot)).astype(float)


def jacobian_central(fun, x: np.ndarray, eps: np.ndarray) -> np.ndarray:
    n = x.size
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps[i]
        fp = fun(x + dx)
        fm = fun(x - dx)
        J[:, i] = (fp - fm) / (2.0 * eps[i])
    return J


class SpacecraftGameElliptic:
    """严格椭圆参考轨道版本：

    - chief/deputy 在惯性系进行两体积分（chief 不受控，deputy 受控）
    - 每个时间步把相对状态投影到 chief 的 LVLH
    - 在 LVLH 上用数值雅可比构造 SDC A(x,t)，做 SDRE/GARE 求解
    """

    def __init__(self, a_c: float, e_c: float, gamma: float = np.sqrt(2.0)):
        self.a = float(a_c)
        self.e = float(e_c)
        self.gamma = float(gamma)

        # Table 1
        self.Q = np.block(
            [
                [np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.eye(3)],
            ]
        )

        R_base = np.eye(3) * 1e13
        self.Rp = R_base
        self.Re = (self.gamma**2) * R_base

        self.inv_Rp = np.linalg.inv(self.Rp)
        self.inv_Re = np.linalg.inv(self.Re)

        self.B = np.zeros((6, 3))
        self.B[3:6, :] = np.eye(3)

        # 数值雅可比步长：位置[km]、速度[km/s]
        self.jac_eps = np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6], dtype=float)

    def solve_game_riccati(self, A: np.ndarray) -> np.ndarray | None:
        S_net_subblock = self.inv_Rp - self.inv_Re
        evals = np.linalg.eigvals(S_net_subblock)
        if np.any(evals <= 0):
            return None

        B_eff = self.B
        R_eff = np.linalg.inv(S_net_subblock)

        try:
            return la.solve_continuous_are(A, B_eff, self.Q, R_eff)
        except Exception:
            return None

    def get_control(self, A: np.ndarray, x_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = self.solve_game_riccati(A)
        if P is None:
            return np.zeros(3), np.zeros(3)

        Px = P @ x_rel
        BTPx = Px[3:6]
        u_p = -self.inv_Rp @ BTPx
        u_e = self.inv_Re @ BTPx
        return u_p, u_e

    def sdc_A(self, chief_r: np.ndarray, chief_v: np.ndarray, x_rel: np.ndarray) -> np.ndarray:
        f = lambda x: rel_xdot_uncontrolled(chief_r, chief_v, x)
        return jacobian_central(f, x_rel, self.jac_eps)

    def dynamics_inertial(self, t: float, y: np.ndarray) -> np.ndarray:
        chief_r = y[0:3]
        chief_v = y[3:6]
        deputy_r = y[6:9]
        deputy_v = y[9:12]

        rho, rho_dot, C_LI, omega_l, _ = rel_from_inertial(chief_r, chief_v, deputy_r, deputy_v)
        x_rel = np.hstack((rho, rho_dot))

        A = self.sdc_A(chief_r, chief_v, x_rel)
        u_p, u_e = self.get_control(A, x_rel)
        u_net_l = u_p + u_e

        a_c = two_body_accel(chief_r)
        a_d = two_body_accel(deputy_r) + (C_LI @ u_net_l)

        return np.hstack((chief_v, a_c, deputy_v, a_d)).astype(float)


if __name__ == "__main__":
    # Table 1 parameters
    a_c = 15000.0
    e_c = 0.5
    gamma = np.sqrt(2.0)

    game = SpacecraftGameElliptic(a_c=a_c, e_c=e_c, gamma=gamma)

    # chief initial state (perigee)
    r_c0, v_c0 = chief_perigee_state(a_c, e_c)

    # pursuer initial relative state in LVLH (evader at origin)
    rho0 = np.array([500.0, 500.0, 500.0], dtype=float)
    rhodot0 = np.array([0.01, 0.01, 0.01], dtype=float)

    r_d0, v_d0 = inertial_from_rel(r_c0, v_c0, rho0, rhodot0)

    y0 = np.hstack((r_c0, v_c0, r_d0, v_d0))

    t_span = (0.0, 10000.0)
    t_eval = np.arange(t_span[0], t_span[1] + 1e-9, 10.0)

    print("开始 SDRE（严格椭圆版）博弈仿真...")
    sol = solve_ivp(game.dynamics_inertial, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-9)
    print("仿真结束.")

    SAVE_GIF = True

    # 惯性系(ECI)下的轨迹：chief/target 与 deputy/pursuer 都在惯性系积分
    fig_eci = plt.figure(figsize=(10, 8))
    ax_eci = plt.axes(projection="3d")
    plot_earth_sphere(ax_eci)
    ax_eci.plot3D(sol.y[0], sol.y[1], sol.y[2], "r", label="Target/Chief (ECI)")
    ax_eci.plot3D(sol.y[6], sol.y[7], sol.y[8], "b", label="Pursuer/Deputy (ECI)")
    ax_eci.scatter3D(sol.y[0, 0], sol.y[1, 0], sol.y[2, 0], c="r", marker="o", s=30)
    ax_eci.scatter3D(sol.y[6, 0], sol.y[7, 0], sol.y[8, 0], c="b", marker="o", s=30)
    ax_eci.set_xlabel("ECI x [km]")
    ax_eci.set_ylabel("ECI y [km]")
    ax_eci.set_zlabel("ECI z [km]")
    ax_eci.set_title("Inertial Trajectories (ECI)")
    ax_eci.legend()
    plt.show()

    if SAVE_GIF:
        chief_r_eci = sol.y[0:3]
        deputy_r_eci = sol.y[6:9]
        save_eci_gif(chief_r_eci, deputy_r_eci, out_path="eci_animation_elliptic.gif", stride=5, fps=20)

    # reconstruct relative LVLH trajectory for plotting
    rel_xyz = np.zeros((3, sol.t.size), dtype=float)
    for k in range(sol.t.size):
        yk = sol.y[:, k]
        chief_r = yk[0:3]
        chief_v = yk[3:6]
        deputy_r = yk[6:9]
        deputy_v = yk[9:12]
        rho, _, _, _, _ = rel_from_inertial(chief_r, chief_v, deputy_r, deputy_v)
        rel_xyz[:, k] = rho

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    ax.plot3D(rel_xyz[0], rel_xyz[1], rel_xyz[2], "b", label="Pursuer (relative LVLH)")
    ax.scatter3D(0, 0, 0, c="r", marker="*", s=100, label="Evader (origin)")
    ax.scatter3D(rho0[0], rho0[1], rho0[2], c="g", marker="o", label="Start")
    ax.set_xlabel("Radial x [km]")
    ax.set_ylabel("Along-track y [km]")
    ax.set_zlabel("Cross-track z [km]")
    ax.set_title("Pursuit-Evasion (Elliptic Chief, LVLH Relative)")
    ax.legend()
    plt.show()

    dist = np.linalg.norm(rel_xyz, axis=0)
    plt.figure()
    plt.plot(sol.t, dist)
    plt.xlabel("Time [s]")
    plt.ylabel("Relative Distance [km]")
    plt.title("Interception Progress")
    plt.grid(True)
    plt.show()
