"""
SDRE 追逃博弈仿真 - poliastro 版本

特性：
- 使用 poliastro 库处理轨道初始化（从 Kepler 根数）
- 使用 poliastro 进行轨道传播（支持椭圆轨道）
- 自动处理坐标系转换（ECI/LVLH）
- 保持 SDRE 博弈控制逻辑不变
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path

from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import zhplot

MU_EARTH = 398600.4418  # km^3/s^2
RE_EARTH = 6378.137  # km
J2 = 1.08262668e-3    # Earth J2


def plot_earth_sphere(ax, radius=RE_EARTH, n_u=36, n_v=18, alpha=0.25):
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="lightgray", linewidth=0, alpha=alpha)


def lvlh_dcm(chief_r: np.ndarray, chief_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (C_LI, C_IL): LVLH <-> Inertial 变换矩阵"""
    ir = chief_r / np.linalg.norm(chief_r)
    h = np.cross(chief_r, chief_v)
    ih = h / np.linalg.norm(h)
    it = np.cross(ih, ir)
    it = it / np.linalg.norm(it)
    C_LI = np.column_stack((ir, it, ih))
    C_IL = C_LI.T
    return C_LI, C_IL


def lvlh_omega(chief_r: np.ndarray, chief_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LVLH 角速度及其导数（惯性系表达）"""
    r = np.linalg.norm(chief_r)
    h = np.cross(chief_r, chief_v)
    omega = h / (r**2)
    rdot = float(np.dot(chief_r, chief_v) / r)
    omega_dot = -2.0 * (rdot / r) * omega
    return omega, omega_dot


def rel_from_inertial(
    chief_r: np.ndarray, chief_v: np.ndarray, deputy_r: np.ndarray, deputy_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """从惯性状态得到 LVLH 相对状态 (rho, rho_dot)"""
    C_LI, C_IL = lvlh_dcm(chief_r, chief_v)
    omega_i, _ = lvlh_omega(chief_r, chief_v)
    omega_l = C_IL @ omega_i

    r_rel_i = deputy_r - chief_r
    v_rel_i = deputy_v - chief_v

    rho = C_IL @ r_rel_i
    rho_dot = C_IL @ v_rel_i - np.cross(omega_l, rho)
    return rho, rho_dot


def inertial_from_rel(
    chief_r: np.ndarray, chief_v: np.ndarray, rho: np.ndarray, rho_dot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """从 LVLH 相对状态得到惯性状态"""
    C_LI, C_IL = lvlh_dcm(chief_r, chief_v)
    omega_i, _ = lvlh_omega(chief_r, chief_v)
    omega_l = C_IL @ omega_i

    deputy_r = chief_r + C_LI @ rho
    v_rel_i = C_LI @ (rho_dot + np.cross(omega_l, rho))
    deputy_v = chief_v + v_rel_i
    return deputy_r, deputy_v


def two_body_accel(r: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """两体引力加速度"""
    return -mu * r / (np.linalg.norm(r) ** 3)


def j2_accel(r: np.ndarray, mu: float = MU_EARTH, Re: float = RE_EARTH, J2_val: float = J2) -> np.ndarray:
    """J2 摄动加速度（单位 km/s^2）。"""
    x, y, z = r
    r_norm = np.linalg.norm(r)
    if r_norm == 0.0:
        return np.zeros(3, dtype=float)
    zx = z / r_norm
    factor = 1.5 * J2_val * mu * (Re**2) / (r_norm**5)
    ax = factor * x * (5 * zx**2 - 1)
    ay = factor * y * (5 * zx**2 - 1)
    az = factor * z * (5 * zx**2 - 3)
    return np.array([ax, ay, az], dtype=float)


def rel_xdot_uncontrolled(chief_r: np.ndarray, chief_v: np.ndarray, x_rel: np.ndarray) -> np.ndarray:
    """LVLH 相对动力学（不含控制）"""
    rho = x_rel[0:3]
    rho_dot = x_rel[3:6]

    deputy_r, deputy_v = inertial_from_rel(chief_r, chief_v, rho, rho_dot)

    a_c = two_body_accel(chief_r)
    a_d = two_body_accel(deputy_r)
    a_rel_i = a_d - a_c

    C_LI, C_IL = lvlh_dcm(chief_r, chief_v)
    omega_i, omega_dot_i = lvlh_omega(chief_r, chief_v)
    omega_l = C_IL @ omega_i
    omega_dot_l = C_IL @ omega_dot_i

    rho_ddot = (
        C_IL @ a_rel_i
        - 2.0 * np.cross(omega_l, rho_dot)
        - np.cross(omega_dot_l, rho)
        - np.cross(omega_l, np.cross(omega_l, rho))
    )
    return np.hstack((rho_dot, rho_ddot)).astype(float)


def jacobian_central(fun, x: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """中心差分数值雅可比"""
    n = x.size
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps[i]
        fp = fun(x + dx)
        fm = fun(x - dx)
        J[:, i] = (fp - fm) / (2.0 * eps[i])
    return J


class SpacecraftGamePoliastro:
    """使用 poliastro 处理轨道的 SDRE 博弈版本"""

    def __init__(self, a_c: float, e_c: float, gamma: float = np.sqrt(2.0)):
        self.a = float(a_c)
        self.e = float(e_c)
        self.gamma = float(gamma)

        # Table 1: Q = diag(I3, I3), R = 1e13 * I3
        self.Q = np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), np.eye(3)]])
        R_base = np.eye(3) * 1e13
        self.Rp = R_base
        self.Re = (self.gamma**2) * R_base
        self.inv_Rp = np.linalg.inv(self.Rp)
        self.inv_Re = np.linalg.inv(self.Re)

        self.B = np.zeros((6, 3))
        self.B[3:6, :] = np.eye(3)

        # 数值雅可比步长
        self.jac_eps = np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6], dtype=float)

    def solve_game_riccati(self, A: np.ndarray) -> np.ndarray | None:
        """求解 GARE"""
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
        """计算 SDRE 控制"""
        P = self.solve_game_riccati(A)
        if P is None:
            return np.zeros(3), np.zeros(3)

        Px = P @ x_rel
        BTPx = Px[3:6]
        u_p = -self.inv_Rp @ BTPx
        u_e = self.inv_Re @ BTPx
        return u_p, u_e

    def sdc_A(self, chief_r: np.ndarray, chief_v: np.ndarray, x_rel: np.ndarray) -> np.ndarray:
        """数值雅可比构造 SDC 矩阵"""
        f = lambda x: rel_xdot_uncontrolled(chief_r, chief_v, x)
        return jacobian_central(f, x_rel, self.jac_eps)

    def dynamics_inertial(self, t: float, y: np.ndarray) -> np.ndarray:
        """惯性系动力学（chief 不受控，deputy 受控）"""
        chief_r = y[0:3]
        chief_v = y[3:6]
        deputy_r = y[6:9]
        deputy_v = y[9:12]

        rho, rho_dot = rel_from_inertial(chief_r, chief_v, deputy_r, deputy_v)
        x_rel = np.hstack((rho, rho_dot))

        A = self.sdc_A(chief_r, chief_v, x_rel)
        u_p, u_e = self.get_control(A, x_rel)
        u_net_l = u_p + u_e

        C_LI, _ = lvlh_dcm(chief_r, chief_v)

        a_c = two_body_accel(chief_r) + j2_accel(chief_r)
        a_d = two_body_accel(deputy_r) + j2_accel(deputy_r) + (C_LI @ u_net_l)

        return np.hstack((chief_v, a_c, deputy_v, a_d)).astype(float)


def save_eci_gif(chief_r_eci, deputy_r_eci, out_path="outputs/gifs/eci_animation_poliastro.gif", stride=5, fps=20):
    """保存 ECI 三维轨迹动图"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(0, chief_r_eci.shape[1], stride, dtype=int)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    plot_earth_sphere(ax)
    ax.set_xlabel("ECI x [km]")
    ax.set_ylabel("ECI y [km]")
    ax.set_zlabel("ECI z [km]")
    ax.set_title("惯性系轨迹（ECI）- 动画")

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


if __name__ == "__main__":
    # Table 1 参数
    a_c = 15000.0  # km
    e_c = 0.5
    gamma = np.sqrt(2.0)

    # 使用 poliastro 创建 chief 轨道（从近地点开始）
    # 注意：poliastro 0.7.0 版本的 API 与后续版本有所不同
    # 这里使用椭圆轨道，近地点位于 x 轴正向
    epoch = Time("2025-01-01 00:00:00", scale="utc")
    
    # 从轨道根数创建 Orbit 对象
    chief_orbit = Orbit.from_classical(
        Earth,
        a_c * u.km,
        e_c * u.one,
        0 * u.deg,  # inclination
        0 * u.deg,  # raan
        0 * u.deg,  # argp
        0 * u.deg,  # nu (true anomaly at epoch - perigee)
        epoch=epoch,
    )

    # 获取初始状态向量 (ECI)
    r_c0 = chief_orbit.r.to(u.km).value  # km
    v_c0 = chief_orbit.v.to(u.km / u.s).value  # km/s

    # Pursuer 初始相对状态（LVLH）
    rho0 = np.array([500.0, 500.0, 500.0], dtype=float)
    rhodot0 = np.array([0.01, 0.01, 0.01], dtype=float)

    # 转换到惯性系
    r_d0, v_d0 = inertial_from_rel(r_c0, v_c0, rho0, rhodot0)

    # 构造初始状态向量
    y0 = np.hstack((r_c0, v_c0, r_d0, v_d0))

    # 初始化博弈
    game = SpacecraftGamePoliastro(a_c=a_c, e_c=e_c, gamma=gamma)

    # 积分求解
    t_span = (0.0, 20000.0)  # 20000 秒
    t_eval = np.arange(t_span[0], t_span[1] + 1e-9, 10.0)

    print("开始 SDRE（poliastro 版）博弈仿真...")
    sol = solve_ivp(game.dynamics_inertial, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-9)
    print("仿真结束.")

    SAVE_GIF = False

    # 先重建 LVLH 相对轨迹，便于计算最近距离
    rel_xyz = np.zeros((3, sol.t.size), dtype=float)
    for k in range(sol.t.size):
        yk = sol.y[:, k]
        chief_r = yk[0:3]
        chief_v = yk[3:6]
        deputy_r = yk[6:9]
        deputy_v = yk[9:12]
        rho, _ = rel_from_inertial(chief_r, chief_v, deputy_r, deputy_v)
        rel_xyz[:, k] = rho

    dist = np.linalg.norm(rel_xyz, axis=0)
    k_min = int(np.argmin(dist))
    min_dist = float(dist[k_min])
    t_min = float(sol.t[k_min])
    print(f"最近距离: {min_dist:.6g} km @ t={t_min:.6g} s")

    # 惯性系 (ECI) 轨迹图
    fig_eci = plt.figure(figsize=(10, 8))
    ax_eci = plt.axes(projection="3d")
    plot_earth_sphere(ax_eci)
    ax_eci.plot3D(sol.y[0], sol.y[1], sol.y[2], "r", label="Target/Chief (ECI)")
    ax_eci.plot3D(sol.y[6], sol.y[7], sol.y[8], "b", label="Pursuer/Deputy (ECI)")
    ax_eci.scatter3D(sol.y[0, 0], sol.y[1, 0], sol.y[2, 0], c="r", marker="o", s=30)
    ax_eci.scatter3D(sol.y[6, 0], sol.y[7, 0], sol.y[8, 0], c="b", marker="o", s=30)
    ax_eci.scatter3D(sol.y[6, k_min], sol.y[7, k_min], sol.y[8, k_min], c="b", marker="x", s=60, label=f"Closest (t={t_min:.0f}s)")
    ax_eci.set_xlabel("ECI x [km]")
    ax_eci.set_ylabel("ECI y [km]")
    ax_eci.set_zlabel("ECI z [km]")
    ax_eci.set_title("惯性系轨迹（ECI）- poliastro 版本")
    ax_eci.legend()
    plt.show()

    if SAVE_GIF:
        chief_r_eci = sol.y[0:3]
        deputy_r_eci = sol.y[6:9]
        save_eci_gif(chief_r_eci, deputy_r_eci, out_path="outputs/gifs/eci_animation_poliastro.gif", stride=5, fps=20)

    # LVLH 相对轨迹
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    ax.plot3D(rel_xyz[0], rel_xyz[1], rel_xyz[2], "b", label="Pursuer (relative LVLH)")
    ax.scatter3D(0, 0, 0, c="r", marker="*", s=100, label="Evader (origin)")
    ax.scatter3D(rho0[0], rho0[1], rho0[2], c="g", marker="o", label="Start")
    ax.scatter3D(rel_xyz[0, k_min], rel_xyz[1, k_min], rel_xyz[2, k_min], c="b", marker="x", s=60, label=f"Closest (t={t_min:.0f}s)")
    ax.set_xlabel("Radial x [km]")
    ax.set_ylabel("Along-track y [km]")
    ax.set_zlabel("Cross-track z [km]")
    ax.set_title("追逃相对轨迹（LVLH）- poliastro 版本")
    ax.legend()
    plt.show()

    # 相对距离随时间
    plt.figure()
    plt.plot(sol.t, dist)
    plt.scatter([t_min], [min_dist], c="b", marker="x", s=60, label="Closest")
    plt.xlabel("Time [s]")
    plt.ylabel("Relative Distance [km]")
    plt.title("接近过程（相对距离）")
    plt.grid(True)
    plt.legend()
    plt.show()
