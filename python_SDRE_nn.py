import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from python_SDRE import SpacecraftGame, plot_earth_sphere, save_eci_gif


class TorchSurrogate:
    def __init__(self, model_path: str):
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        self.payload = payload

        model_cfg = payload["model"]
        self.model_type = model_cfg.get("type", "u")  # u | P
        in_dim = int(model_cfg["in_dim"])
        out_dim = int(model_cfg["out_dim"])
        hidden = int(model_cfg["hidden"])
        layers = int(model_cfg["layers"])

        blocks = []
        last = in_dim
        for _ in range(layers):
            blocks.append(torch.nn.Linear(last, hidden))
            blocks.append(torch.nn.Tanh())
            last = hidden
        blocks.append(torch.nn.Linear(last, out_dim))
        self.net = torch.nn.Sequential(*blocks)

        state_dict = payload["state_dict"]
        # Training script saves keys like "net.0.weight"; adapt to bare Sequential keys.
        if any(k.startswith("net.") for k in state_dict.keys()):
            state_dict = {k[len("net."):]: v for k, v in state_dict.items()}

        self.net.load_state_dict(state_dict)
        self.net.eval()

        norm = payload["norm"]
        self.x_mean = torch.tensor(norm["x_mean"], dtype=torch.float32)
        self.x_std = torch.tensor(norm["x_std"], dtype=torch.float32)
        self.y_mean = torch.tensor(norm["y_mean"], dtype=torch.float32)
        self.y_std = torch.tensor(norm["y_std"], dtype=torch.float32)

        # Training config may include whether the teacher target is u_net or u_p.
        self.target = payload.get("config", {}).get("target", "u_net")

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        xt = torch.tensor(x, dtype=torch.float32)
        xn = (xt - self.x_mean) / self.x_std
        yn = self.net(xn)
        y = yn * self.y_std + self.y_mean
        return y.detach().cpu().numpy()

    def control_from_output(self, game: SpacecraftGame, state: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Convert network output into an LVLH acceleration control vector (3,)."""
        if y.shape[0] == 3:
            return y
        if y.shape[0] == 21:
            # Upper-triangular symmetric 6x6 -> P
            P = np.zeros((6, 6), dtype=float)
            k = 0
            for i in range(6):
                for j in range(i, 6):
                    P[i, j] = float(y[k])
                    P[j, i] = float(y[k])
                    k += 1

            # u_p = -inv(Rp) * B^T P x , with B^T selecting velocity rows.
            x = state
            Px = P @ x
            BTPx = Px[3:6]
            u_p = -game.inv_Rp @ BTPx
            if self.target == "u_p":
                return u_p

            # u_net = u_p + u_e, where u_e = inv(Re) * B^T P x
            u_e = game.inv_Re @ BTPx
            return u_p + u_e

        raise ValueError(f"Unsupported network output dim: {y.shape}")


def teacher_control(game: SpacecraftGame, state: np.ndarray, target: str) -> np.ndarray:
    A, Bp, Be = game.get_sdc_matrices(state)
    P = game.solve_game_riccati(A, Bp, Be)
    if P is None:
        return np.zeros(3, dtype=float)
    Px = P @ state
    BTPx = Px[3:6]
    u_p = -game.inv_Rp @ BTPx
    u_e = game.inv_Re @ BTPx
    if target == "u_p":
        return u_p
    if target == "u_net":
        return u_p + u_e
    raise ValueError(f"Unknown target: {target}")


def simulate_with_nn(
    game: SpacecraftGame,
    surrogate: TorchSurrogate,
    x0: np.ndarray,
    t_span,
    t_eval,
    *,
    target: str,
    max_step: float | None = None,
):
    Rc = game.Rc
    n = game.n
    mu = 398600.4418

    def dynamics(t, state):
        x, y, z, vx, vy, vz = state
        y_out = surrogate.predict(state)
        u = surrogate.control_from_output(game, state, y_out)

        r_d = np.sqrt((Rc + x) ** 2 + y**2 + z**2)
        ax = 2 * n * vy + n**2 * x + mu / Rc**2 - mu * (Rc + x) / r_d**3 + u[0]
        ay = -2 * n * vx + n**2 * y - mu * y / r_d**3 + u[1]
        az = -mu * z / r_d**3 + u[2]
        return np.array([vx, vy, vz, ax, ay, az], dtype=float)

    return solve_ivp(dynamics, t_span, x0, t_eval=t_eval, rtol=1e-6, atol=1e-9, max_step=max_step)


def simulate_with_teacher(
    game: SpacecraftGame,
    x0: np.ndarray,
    t_span,
    t_eval,
    *,
    target: str,
    max_step: float | None = None,
):
    Rc = game.Rc
    n = game.n
    mu = 398600.4418

    def dynamics(t, state):
        x, y, z, vx, vy, vz = state
        u = teacher_control(game, state, target=target)
        r_d = np.sqrt((Rc + x) ** 2 + y**2 + z**2)
        ax = 2 * n * vy + n**2 * x + mu / Rc**2 - mu * (Rc + x) / r_d**3 + u[0]
        ay = -2 * n * vx + n**2 * y - mu * y / r_d**3 + u[1]
        az = -mu * z / r_d**3 + u[2]
        return np.array([vx, vy, vz, ax, ay, az], dtype=float)

    return solve_ivp(dynamics, t_span, x0, t_eval=t_eval, rtol=1e-6, atol=1e-9, max_step=max_step)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def rollout_controls(game: SpacecraftGame, surrogate: TorchSurrogate, states: np.ndarray, *, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_teacher, u_nn) evaluated on the same state sequence."""
    u_t = np.zeros((states.shape[0], 3), dtype=float)
    u_n = np.zeros((states.shape[0], 3), dtype=float)
    for k in range(states.shape[0]):
        x = states[k]
        u_t[k] = teacher_control(game, x, target=target)
        y_out = surrogate.predict(x)
        u_n[k] = surrogate.control_from_output(game, x, y_out)
    return u_t, u_n


def main():
    parser = argparse.ArgumentParser(description="Circular LVLH SDRE simulation with a torch surrogate control law")
    parser.add_argument("--model", type=str, default="sdre_control_net.pt", help="Path to trained surrogate model")
    parser.add_argument("--tf", type=float, default=10000.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting (useful for long comparisons)")
    parser.add_argument("--compare-teacher", action="store_true", help="Also simulate online SDRE teacher and compare")
    parser.add_argument("--target", type=str, default=None, choices=["u_net", "u_p"], help="Override control target")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train it with train_control_surrogate_torch.py")

    game = SpacecraftGame(chief_semi_major_axis=15000.0, chief_eccentricity=0.5, gamma=np.sqrt(2.0))
    surrogate = TorchSurrogate(str(model_path))

    target = args.target or surrogate.target or "u_net"

    x0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01], dtype=float)
    t_span = (0.0, float(args.tf))
    t_eval = np.arange(t_span[0], t_span[1] + 1e-9, float(args.dt))
    max_step = float(args.dt)

    print(f"开始 NN 近似控制律仿真（圆轨道 LVLH, target={target}）...")
    sol_nn = simulate_with_nn(game, surrogate, x0, t_span, t_eval, target=target, max_step=max_step)
    print("NN 仿真结束.")

    sol_teacher = None
    if args.compare_teacher:
        print("开始 teacher (在线 SDRE/GARE) 仿真...")
        sol_teacher = simulate_with_teacher(game, x0, t_span, t_eval, target=target, max_step=max_step)
        print("teacher 仿真结束.")

        x_rmse = rmse(sol_nn.y.T, sol_teacher.y.T)
        dist_nn = np.sqrt(sol_nn.y[0] ** 2 + sol_nn.y[1] ** 2 + sol_nn.y[2] ** 2)
        dist_teacher = np.sqrt(sol_teacher.y[0] ** 2 + sol_teacher.y[1] ** 2 + sol_teacher.y[2] ** 2)
        dist_rmse = rmse(dist_nn, dist_teacher)

        # Policy error: evaluate both controls on the teacher trajectory
        u_t, u_n = rollout_controls(game, surrogate, sol_teacher.y.T, target=target)
        u_rmse = rmse(u_n, u_t)
        print(
            f"对比指标: state_RMSE={x_rmse:.6g}, dist_RMSE={dist_rmse:.6g}, "
            f"u_RMSE(on teacher traj)={u_rmse:.6g}, final_dist_nn={dist_nn[-1]:.6g}, final_dist_teacher={dist_teacher[-1]:.6g}"
        )

    if not args.no_plots:
        # LVLH relative 3D
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection="3d")
        ax.plot3D(sol_nn.y[0], sol_nn.y[1], sol_nn.y[2], "b", label="NN (relative LVLH)")
        if sol_teacher is not None:
            ax.plot3D(sol_teacher.y[0], sol_teacher.y[1], sol_teacher.y[2], "k--", label="Teacher SDRE (relative LVLH)")
        ax.scatter3D(0, 0, 0, c="r", marker="*", s=100, label="Evader (origin)")
        ax.scatter3D(x0[0], x0[1], x0[2], c="g", marker="o", label="Start")
        ax.set_xlabel("Radial x [km]")
        ax.set_ylabel("Along-track y [km]")
        ax.set_zlabel("Cross-track z [km]")
        ax.set_title("LVLH Relative Trajectory")
        ax.legend()
        plt.show()

    # Build ideal chief ECI circle, map LVLH rho -> ECI
    Rc = game.Rc
    n = game.n
    theta = n * sol_nn.t
    chief_r_eci = np.vstack((Rc * np.cos(theta), Rc * np.sin(theta), np.zeros_like(theta)))
    chief_v_eci = np.vstack((-Rc * n * np.sin(theta), Rc * n * np.cos(theta), np.zeros_like(theta)))

    pursuer_r_eci = np.zeros_like(chief_r_eci)
    for k in range(sol_nn.t.size):
        r = chief_r_eci[:, k]
        v = chief_v_eci[:, k]
        ir = r / np.linalg.norm(r)
        h = np.cross(r, v)
        ih = h / np.linalg.norm(h)
        it = np.cross(ih, ir)
        it = it / np.linalg.norm(it)
        C_LI = np.column_stack((ir, it, ih))
        rho = sol_nn.y[0:3, k]
        pursuer_r_eci[:, k] = r + C_LI @ rho

    pursuer_r_eci_teacher = None
    if sol_teacher is not None:
        pursuer_r_eci_teacher = np.zeros_like(chief_r_eci)
        for k in range(sol_teacher.t.size):
            r = chief_r_eci[:, k]
            v = chief_v_eci[:, k]
            ir = r / np.linalg.norm(r)
            h = np.cross(r, v)
            ih = h / np.linalg.norm(h)
            it = np.cross(ih, ir)
            it = it / np.linalg.norm(it)
            C_LI = np.column_stack((ir, it, ih))
            rho = sol_teacher.y[0:3, k]
            pursuer_r_eci_teacher[:, k] = r + C_LI @ rho

    if not args.no_plots:
        plt.figure(figsize=(10, 8))
        ax_eci = plt.axes(projection="3d")
        plot_earth_sphere(ax_eci)
        ax_eci.plot3D(chief_r_eci[0], chief_r_eci[1], chief_r_eci[2], "r", label="Target/Chief (ECI)")
        ax_eci.plot3D(pursuer_r_eci[0], pursuer_r_eci[1], pursuer_r_eci[2], "b", label="NN (ECI)")
        if pursuer_r_eci_teacher is not None:
            ax_eci.plot3D(pursuer_r_eci_teacher[0], pursuer_r_eci_teacher[1], pursuer_r_eci_teacher[2], "k--", label="Teacher SDRE (ECI)")
        ax_eci.set_xlabel("ECI x [km]")
        ax_eci.set_ylabel("ECI y [km]")
        ax_eci.set_zlabel("ECI z [km]")
        ax_eci.set_title("Inertial Trajectories (ECI)")
        ax_eci.legend()
        plt.show()

    if args.save_gif:
        save_eci_gif(chief_r_eci, pursuer_r_eci, out_path="eci_animation_nn.gif", stride=5, fps=20)

    dist = np.sqrt(sol_nn.y[0] ** 2 + sol_nn.y[1] ** 2 + sol_nn.y[2] ** 2)
    if not args.no_plots:
        plt.figure()
        plt.plot(sol_nn.t, dist, label="NN")
        if sol_teacher is not None:
            dist_t = np.sqrt(sol_teacher.y[0] ** 2 + sol_teacher.y[1] ** 2 + sol_teacher.y[2] ** 2)
            plt.plot(sol_teacher.t, dist_t, "k--", label="Teacher")
        plt.xlabel("Time [s]")
        plt.ylabel("Relative Distance [km]")
        plt.title("Interception Progress")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
