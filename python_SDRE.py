import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path

# ==========================================
# 物理常数与参数定义
# ==========================================
MU_EARTH = 398600.4418  # 地球引力常数 [km^3/s^2]
RE_EARTH = 6378.137     # 地球半径 [km]


def plot_earth_sphere(ax, radius=RE_EARTH, n_u=36, n_v=18, alpha=0.25):
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="lightgray", linewidth=0, alpha=alpha)


def save_eci_gif(chief_r_eci, pursuer_r_eci, out_path="outputs/gifs/eci_animation.gif", stride=5, fps=20):
    """保存ECI三维轨迹动图（GIF）。"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(0, chief_r_eci.shape[1], stride, dtype=int)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    plot_earth_sphere(ax)
    ax.set_xlabel("ECI x [km]")
    ax.set_ylabel("ECI y [km]")
    ax.set_zlabel("ECI z [km]")
    ax.set_title("Inertial Trajectories (ECI) - Animation")

    (chief_line,) = ax.plot([], [], [], "r", label="Target/Chief (ECI)")
    (pursuer_line,) = ax.plot([], [], [], "b", label="Pursuer/Deputy (ECI)")
    chief_pt = ax.scatter([], [], [], c="r", s=20)
    pursuer_pt = ax.scatter([], [], [], c="b", s=20)
    ax.legend()

    all_pts = np.hstack((chief_r_eci, pursuer_r_eci))
    max_range = np.max(np.ptp(all_pts, axis=1))
    center = np.mean(all_pts, axis=1)
    half = 0.5 * max_range
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    def init():
        chief_line.set_data([], [])
        chief_line.set_3d_properties([])
        pursuer_line.set_data([], [])
        pursuer_line.set_3d_properties([])
        return (chief_line, pursuer_line, chief_pt, pursuer_pt)

    def update(frame_i):
        k = idx[frame_i]
        chief_line.set_data(chief_r_eci[0, idx[: frame_i + 1]], chief_r_eci[1, idx[: frame_i + 1]])
        chief_line.set_3d_properties(chief_r_eci[2, idx[: frame_i + 1]])
        pursuer_line.set_data(pursuer_r_eci[0, idx[: frame_i + 1]], pursuer_r_eci[1, idx[: frame_i + 1]])
        pursuer_line.set_3d_properties(pursuer_r_eci[2, idx[: frame_i + 1]])

        # 3D scatter update via private API (matplotlib limitation)
        chief_pt._offsets3d = (
            np.array([chief_r_eci[0, k]]),
            np.array([chief_r_eci[1, k]]),
            np.array([chief_r_eci[2, k]]),
        )
        pursuer_pt._offsets3d = (
            np.array([pursuer_r_eci[0, k]]),
            np.array([pursuer_r_eci[1, k]]),
            np.array([pursuer_r_eci[2, k]]),
        )
        return (chief_line, pursuer_line, chief_pt, pursuer_pt)

    ani = animation.FuncAnimation(fig, update, frames=idx.size, init_func=init, interval=1000 / fps, blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

class SpacecraftGame:
    def __init__(self, chief_semi_major_axis, chief_eccentricity=0.0, gamma=np.sqrt(2.0)):
        """
        初始化博弈环境
        :param chief_semi_major_axis: 主星（目标/逃逸者）半长轴 a_c [km]
        :param chief_eccentricity: 主星偏心率 e_c（当前模型使用常值 n，仅存储该值）
        :param gamma: 控制缩放因子（用于构造追/逃的相对控制代价）
        """
        self.a = float(chief_semi_major_axis)
        self.e = float(chief_eccentricity)
        self.gamma = float(gamma)

        # 当前动力学以圆轨道 LVLH 为基础，使用常值平均角速度 n≈sqrt(mu/a^3)
        self.Rc = self.a
        self.n = np.sqrt(MU_EARTH / self.a**3)  # Mean Motion
        
        # 定义博弈权重矩阵
        # Q: 状态惩罚矩阵 (希望相对距离和速度为0)
        # 较大的Q值意味着追踪者更急切地想要减小误差
        # Table 1: Q = diag(I3, I3)
        self.Q = np.block([
            [np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.eye(3)],
        ])
        
        # Table 1: R = I3 * 1e13
        # 为了满足鞍点存在条件 inv(Rp) > inv(Re)，这里采用 Re = gamma^2 * R
        # 则 inv(Rp) - inv(Re) = (1 - 1/gamma^2) * inv(R) > 0
        R_base = np.eye(3) * 1e13

        # Rp: 追踪者控制权重
        self.Rp = R_base
        self.inv_Rp = np.linalg.inv(self.Rp)
        
        # Re: 逃逸者控制权重
        self.Re = (self.gamma**2) * R_base
        self.inv_Re = np.linalg.inv(self.Re)

    def get_sdc_matrices(self, state):
        """
        计算状态依赖系数矩阵 A(x)
        采用 'Apparent Linearization' (即雅可比矩阵) 作为 SDC 的一种鲁棒近似形式。
        这种形式在原点附近非奇异，且能很好地捕捉局部非线性刚度变化。
        
        State x = [x, y, z, vx, vy, vz]
        """
        x, y, z, vx, vy, vz = state
        
        # 计算追踪者的地心距离 Rd
        # 在LVLH系中，Rd = sqrt((Rc + x)^2 + y^2 + z^2)
        Rd_sq = (self.Rc + x)**2 + y**2 + z**2
        Rd = np.sqrt(Rd_sq)
        
        # 构造 A(x) 矩阵 (6x6)
        # A(x) = [   0      I   ]
        #        [ A_grav  A_cor]
        
        A = np.zeros((6, 6))
        
        # 运动学部分: r_dot = v
        A[0:3, 3:6] = np.eye(3)
        
        # 动力学部分: v_dot = f(r, v)
        # 科里奥利项 (线性部分)
        A[3, 4] = 2 * self.n
        A[4, 3] = -2 * self.n
        
        # 非线性引力梯度项 + 离心力项
        # 我们需要计算非线性函数 N(x) = -mu/Rd^3 * + 关于位置的雅可比
        # 这种线性化保留了随状态变化的刚度特性 (Gravity Gradient Stiffness)
        
        mu = MU_EARTH
        Rc = self.Rc
        n_sq = self.n**2
        
        # 辅助变量: 偏导数
        # d(Rd)/dx = (Rc+x)/Rd
        dRd_dx = (Rc + x) / Rd
        dRd_dy = y / Rd
        dRd_dz = z / Rd
        
        term1 = -mu / Rd**3
        term2 = 3 * mu / Rd**4 # 来源于 d(1/Rd^3)/dRd = -3/Rd^4
        
        # --- 填充 A_grav 子矩阵 (3x3) ---
        
        # Row 3 (x-axis dynamics): n^2*x + term1*(Rc+x) + mu/Rc^2 (Constant removed in linearization dynamics)
        # 注意：SDRE用于调节问题，原点必须是平衡点。
        # 在 x=y=z=0 处，动力学平衡，因此常数项被抵消。
        # 下面的矩阵元素代表 "刚度" 变化
        
        # d(ax)/dx, d(ax)/dy, d(ax)/dz
        A[3, 0] = n_sq + term1 + term2 * dRd_dx * (Rc + x)
        A[3, 1] = term2 * dRd_dy * (Rc + x)
        A[3, 2] = term2 * dRd_dz * (Rc + x)
        
        # Row 4 (y-axis dynamics): n^2*y + term1*y
        # d(ay)/dx, d(ay)/dy, d(ay)/dz
        A[4, 0] = term2 * dRd_dx * y
        A[4, 1] = n_sq + term1 + term2 * dRd_dy * y
        A[4, 2] = term2 * dRd_dz * y
        
        # Row 5 (z-axis dynamics): term1*z
        # d(az)/dx, d(az)/dy, d(az)/dz
        A[5, 0] = term2 * dRd_dx * z
        A[5, 1] = term2 * dRd_dy * z
        A[5, 2] = term1 + term2 * dRd_dz * z
        
        # B 矩阵 (输入矩阵)
        # Bp 和 Be 结构相同，只是作用于速度态
        B_block = np.zeros((6, 3))
        B_block[3:6, :] = np.eye(3)
        
        return A, B_block, B_block

    def solve_game_riccati(self, A, Bp, Be):
        """
        求解博弈代数黎卡提方程 (GARE)
        """
        # 1. 计算净控制效能矩阵 S_net = Sp - Se
        # Sp = Bp * inv(Rp) * Bp.T
        # Se = Be * inv(Re) * Be.T
        
        # 由于 Bp = Be = [0; I], 我们可以简化计算
        # S_net 实际上只在右下角 3x3 块非零
        S_net_subblock = self.inv_Rp - self.inv_Re
        
        # 2. 检查正定性 (Saddle Point Existence Condition)
        # 如果追踪者不够强 (inv_Rp <= inv_Re)，则 S_net 不定或负定，ARE无正定解
        evals = np.linalg.eigvals(S_net_subblock)
        if np.any(evals <= 0):
            # 在实际工程中，这代表逃逸者可能逃脱，需要切换策略或报错
            # 这里我们做一个简单的正则化，强制求解器工作（仅用于演示）
            # 或者返回 None 表示无法计算
            return None

        # 3. 构造等效的 LQR 参数以利用 scipy.linalg.solve_continuous_are
        # 标准 ARE: A.T*X + X*A - X*B*inv(R)*B.T*X + Q = 0
        # 我们的 GARE: A.T*P + P*A - P * S_net * P + Q = 0
        # 映射关系: Let B_eff = I (6x6), inv(R_eff) = S_net
        # 因此 R_eff = inv(S_net)
        
        # 构造全尺寸的 S_net
        S_net_full = np.zeros((6, 6))
        S_net_full[3:6, 3:6] = S_net_subblock
        
        # 此时 S_net_full 是奇异的（秩为3），直接求逆不行。
        # 但是 solve_continuous_are 需要 R 是非奇异的。
        # 我们可以利用 solve_continuous_are 的形式：它求解的是 B*inv(R)*B.T
        # 我们可以令 B_eff = [0; I] (6x3), R_eff = inv(S_net_subblock) (3x3)
        # 这样 B_eff * inv(R_eff) * B_eff.T 正好等于 S_net_full
        
        B_eff = np.zeros((6, 3))
        B_eff[3:6, :] = np.eye(3)
        R_eff = np.linalg.inv(S_net_subblock)
        
        try:
            # 求解 ARE
            P = la.solve_continuous_are(A, B_eff, self.Q, R_eff)
            return P
        except Exception as e:
            # 求解失败（例如哈密顿矩阵特征值在虚轴上）
            return None

    def get_control(self, t, state):
        """
        计算当前的双方最优控制律
        """
        A, Bp, Be = self.get_sdc_matrices(state)
        P = self.solve_game_riccati(A, Bp, Be)
        
        if P is None:
            return np.zeros(3), np.zeros(3)
        
        # 计算反馈控制律
        # u_p = -inv(Rp) * Bp.T * P * x
        # u_e = +inv(Re) * Be.T * P * x
        
        # Bp.T * P * x 相当于取 P * x 的后三行
        Px = P @ state
        B_transpose_Px = Px[3:6]
        
        u_p = -self.inv_Rp @ B_transpose_Px
        u_e =  self.inv_Re @ B_transpose_Px
        
        return u_p, u_e

    def dynamics(self, t, state):
        """
        真实非线性动力学方程 (用于积分推演)
        """
        x, y, z, vx, vy, vz = state
        
        # 获取当前双方控制输入
        u_p, u_e = self.get_control(t, state)
        
        # 计算非线性引力加速度
        # r_d 模长
        r_d = np.sqrt((self.Rc + x)**2 + y**2 + z**2)
        mu = MU_EARTH
        n = self.n
        Rc = self.Rc
        
        # 非线性相对加速度 (Euler-Hill Full Nonlinear Equations)
        # ax = 2*n*vy + n^2*x + mu/Rc^2 - mu*(Rc+x)/rd^3 + ux_net
        # ay = -2*n*vx + n^2*y - mu*y/rd^3 + uy_net
        # az = -mu*z/rd^3 + uz_net
        
        u_net = u_p + u_e
        # 在博弈定义中，系统方程是 x_dot = f(x) + Bp*up + Be*ue
        # 我们的 get_control 返回的是绝对值矢量。
        # 逃逸者试图最大化J，其最优控制 u_e = R_e^-1 B_e^T P x。
        # 它作用在动力学上是正向的干扰。
        
        ax = 2*n*vy + n**2*x + mu/Rc**2 - mu*(Rc+x)/r_d**3 + u_net[0]
        ay = -2*n*vx + n**2*y - mu*y/r_d**3 + u_net[1]
        az = -mu*z/r_d**3 + u_net[2]

        return np.array([vx, vy, vz, ax, ay, az], dtype=float)

# ==========================================
# 仿真执行主程序
# ==========================================

if __name__ == "__main__":
    # 1. 设置仿真场景
    # Table 1
    game = SpacecraftGame(chief_semi_major_axis=15000.0, chief_eccentricity=0.5, gamma=np.sqrt(2.0))
    
    # 2. 初始条件
    # 追踪者位于目标后方 10km, 下方 1km, 具有微小的相对速度
    x0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01])
    
    # 3. 积分求解
    t_span = (0, 10000) # 仿真 10000 秒
    t_eval = np.arange(t_span[0], t_span[1] + 1e-9, 10.0)
    
    print("开始 SDRE 博弈仿真...")
    sol = solve_ivp(game.dynamics, t_span, x0, t_eval=t_eval, rtol=1e-5, atol=1e-8)
    print("仿真结束.")

    SAVE_GIF = False  # 是否保存 ECI 轨迹动图
    
    # 4. 结果可视化
    plt.figure(figsize=(10, 8))
    
    # 3D 轨迹图
    ax = plt.axes(projection='3d')
    ax.plot3D(sol.y[0], sol.y[1], sol.y[2], 'b', label='Pursuer Trajectory')
    ax.scatter3D(0, 0, 0, c='r', marker='*', s=100, label='Evader (Target)')
    
    # 起点标记
    ax.scatter3D(x0[0], x0[1], x0[2], c='g', marker='o', label='Start')
    
    ax.set_xlabel('Radial (x) [km]')
    ax.set_ylabel('Along-Track (y) [km]')
    ax.set_zlabel('Cross-Track (z) [km]')
    ax.set_title('Spacecraft Pursuit-Evasion Trajectory (LVLH Frame)')
    ax.legend()
    plt.show()

    # 惯性系(ECI)下的轨迹：target/chief 作为圆轨道航天器随时间运动
    # 说明：本脚本的动力学是在 LVLH 相对系下积分得到 rho(t)。这里用理想圆轨道生成 chief 的 ECI 状态，
    # 并用该时刻的 LVLH 方向余弦矩阵把 rho(t) 转回 ECI，得到 pursuer 的 ECI 轨迹。
    Rc = game.Rc
    n = game.n
    theta = n * sol.t
    chief_r_eci = np.vstack((Rc * np.cos(theta), Rc * np.sin(theta), np.zeros_like(theta)))
    chief_v_eci = np.vstack((-Rc * n * np.sin(theta), Rc * n * np.cos(theta), np.zeros_like(theta)))

    pursuer_r_eci = np.zeros_like(chief_r_eci)
    for k in range(sol.t.size):
        r = chief_r_eci[:, k]
        v = chief_v_eci[:, k]
        ir = r / np.linalg.norm(r)
        h = np.cross(r, v)
        ih = h / np.linalg.norm(h)
        it = np.cross(ih, ir)
        it = it / np.linalg.norm(it)
        C_LI = np.column_stack((ir, it, ih))  # LVLH -> ECI
        rho = sol.y[0:3, k]
        pursuer_r_eci[:, k] = r + C_LI @ rho

    plt.figure(figsize=(10, 8))
    ax_eci = plt.axes(projection='3d')
    plot_earth_sphere(ax_eci)
    ax_eci.plot3D(chief_r_eci[0], chief_r_eci[1], chief_r_eci[2], 'r', label='Target/Chief (ECI)')
    ax_eci.plot3D(pursuer_r_eci[0], pursuer_r_eci[1], pursuer_r_eci[2], 'b', label='Pursuer/Deputy (ECI)')
    ax_eci.scatter3D(chief_r_eci[0, 0], chief_r_eci[1, 0], chief_r_eci[2, 0], c='r', marker='o', s=30)
    ax_eci.scatter3D(pursuer_r_eci[0, 0], pursuer_r_eci[1, 0], pursuer_r_eci[2, 0], c='b', marker='o', s=30)
    ax_eci.set_xlabel('ECI x [km]')
    ax_eci.set_ylabel('ECI y [km]')
    ax_eci.set_zlabel('ECI z [km]')
    ax_eci.set_title('Inertial Trajectories (ECI)')
    ax_eci.legend()
    plt.show()

    if SAVE_GIF:
        save_eci_gif(chief_r_eci, pursuer_r_eci, out_path="outputs/gifs/eci_animation.gif", stride=5, fps=20)
    
    # 相对距离随时间变化
    dist = np.sqrt(sol.y[0]**2 + sol.y[1]**2 + sol.y[2]**2)
    plt.figure()
    plt.plot(sol.t, dist)
    plt.xlabel('Time [s]')
    plt.ylabel('Relative Distance [km]')
    plt.title('Interception Progress')
    plt.grid(True)
    plt.show()