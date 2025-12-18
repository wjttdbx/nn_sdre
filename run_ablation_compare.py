import argparse
import csv
import time
from pathlib import Path

import numpy as np

import torch

from python_SDRE import SpacecraftGame


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


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


class TorchSurrogate:
    def __init__(self, model_path: str):
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        self.payload = payload

        model_cfg = payload["model"]
        self.model_type = model_cfg.get("type", "u")
        self.target = payload.get("config", {}).get("target", "u_net")

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
        if any(k.startswith("net.") for k in state_dict.keys()):
            state_dict = {k[len("net."):]: v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict)
        self.net.eval()

        norm = payload["norm"]
        self.x_mean = torch.tensor(norm["x_mean"], dtype=torch.float32)
        self.x_std = torch.tensor(norm["x_std"], dtype=torch.float32)
        self.y_mean = torch.tensor(norm["y_mean"], dtype=torch.float32)
        self.y_std = torch.tensor(norm["y_std"], dtype=torch.float32)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        xt = torch.tensor(x, dtype=torch.float32)
        xn = (xt - self.x_mean) / self.x_std
        yn = self.net(xn)
        y = yn * self.y_std + self.y_mean
        return y.detach().cpu().numpy()

    def control_from_output(self, game: SpacecraftGame, state: np.ndarray, y: np.ndarray) -> np.ndarray:
        if y.shape[0] == 3:
            return y
        if y.shape[0] == 21:
            P = np.zeros((6, 6), dtype=float)
            k = 0
            for i in range(6):
                for j in range(i, 6):
                    P[i, j] = float(y[k])
                    P[j, i] = float(y[k])
                    k += 1

            Px = P @ state
            BTPx = Px[3:6]
            u_p = -game.inv_Rp @ BTPx
            if self.target == "u_p":
                return u_p
            u_e = game.inv_Re @ BTPx
            return u_p + u_e
        raise ValueError(f"Unsupported network output dim: {y.shape}")

    def control(self, game: SpacecraftGame, state: np.ndarray) -> np.ndarray:
        """Return LVLH acceleration control (3,) for the given state."""
        if self.model_type != "V":
            return self.control_from_output(game, state, self.predict(state))

        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        x_norm = (x - self.x_mean) / self.x_std
        x_norm = x_norm.clone().detach().requires_grad_(True)
        V = self.net(x_norm).sum()
        grad_norm = torch.autograd.grad(V, x_norm, create_graph=False)[0]
        grad = grad_norm / self.x_std
        grad_v = grad[0, 3:6].detach().cpu().numpy()

        u_p = -0.5 * (game.inv_Rp @ grad_v)
        u_e = +0.5 * (game.inv_Re @ grad_v)
        if self.target == "u_p":
            return u_p
        return u_p + u_e


def rel_dynamics(game: SpacecraftGame, state: np.ndarray, u: np.ndarray) -> np.ndarray:
    Rc = game.Rc
    n = game.n
    mu = 398600.4418
    x, y, z, vx, vy, vz = state
    r_d = np.sqrt((Rc + x) ** 2 + y**2 + z**2)
    ax = 2 * n * vy + n**2 * x + mu / Rc**2 - mu * (Rc + x) / r_d**3 + u[0]
    ay = -2 * n * vx + n**2 * y - mu * y / r_d**3 + u[1]
    az = -mu * z / r_d**3 + u[2]
    return np.array([vx, vy, vz, ax, ay, az], dtype=float)


def rollout_fixed_step(policy_fn, game: SpacecraftGame, x0: np.ndarray, tf: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    steps = int(np.floor(tf / dt))
    t = np.linspace(0.0, steps * dt, steps + 1)
    x = np.zeros((steps + 1, 6), dtype=float)
    x[0] = x0

    for k in range(steps):
        xk = x[k]
        u = policy_fn(xk)

        # RK4 with zero-order hold on control u over the step
        k1 = rel_dynamics(game, xk, u)
        k2 = rel_dynamics(game, xk + 0.5 * dt * k1, u)
        k3 = rel_dynamics(game, xk + 0.5 * dt * k2, u)
        k4 = rel_dynamics(game, xk + dt * k3, u)
        x[k + 1] = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, x


def summarize_one(model_path: Path, tf: float, dt: float, target_override: str | None) -> dict:
    game = SpacecraftGame(chief_semi_major_axis=15000.0, chief_eccentricity=0.5, gamma=np.sqrt(2.0))
    surrogate = TorchSurrogate(str(model_path))

    target = target_override or surrogate.target or "u_net"

    x0 = np.array([500.0, 500.0, 500.0, 0.01, 0.01, 0.01], dtype=float)
    t0 = time.time()
    t_nn, x_nn = rollout_fixed_step(lambda s: surrogate.control(game, s), game, x0, tf=tf, dt=dt)
    t1 = time.time()
    t_teacher, x_teacher = rollout_fixed_step(lambda s: teacher_control(game, s, target=target), game, x0, tf=tf, dt=dt)
    t2 = time.time()

    if t_nn.shape != t_teacher.shape or not np.allclose(t_nn, t_teacher):
        raise RuntimeError("Time grids differ unexpectedly")

    state_rmse = rmse(x_nn, x_teacher)

    dist_nn = np.sqrt(x_nn[:, 0] ** 2 + x_nn[:, 1] ** 2 + x_nn[:, 2] ** 2)
    dist_teacher = np.sqrt(x_teacher[:, 0] ** 2 + x_teacher[:, 1] ** 2 + x_teacher[:, 2] ** 2)
    dist_rmse = rmse(dist_nn, dist_teacher)

    # Policy error on teacher trajectory
    u_t = np.zeros((x_teacher.shape[0], 3), dtype=float)
    u_n = np.zeros((x_teacher.shape[0], 3), dtype=float)
    for k in range(x_teacher.shape[0]):
        s = x_teacher[k]
        u_t[k] = teacher_control(game, s, target=target)
        u_n[k] = surrogate.control(game, s)
    u_rmse = rmse(u_n, u_t)

    payload = surrogate.payload
    model_type = payload.get("model", {}).get("type", "u")
    lambda_are = payload.get("config", {}).get("lambda_are", 0.0)

    return {
        "model": str(model_path),
        "model_type": str(model_type),
        "lambda_are": float(lambda_are),
        "target": str(target),
        "tf": float(tf),
        "dt": float(dt),
        "state_rmse": float(state_rmse),
        "dist_rmse": float(dist_rmse),
        "u_rmse_on_teacher_traj": float(u_rmse),
        "final_dist_nn": float(dist_nn[-1]),
        "final_dist_teacher": float(dist_teacher[-1]),
        "seconds_nn": float(t1 - t0),
        "seconds_teacher": float(t2 - t1),
    }


def print_table(rows: list[dict]) -> None:
    cols = [
        "model",
        "model_type",
        "lambda_are",
        "target",
        "state_rmse",
        "dist_rmse",
        "u_rmse_on_teacher_traj",
        "final_dist_nn",
        "final_dist_teacher",
        "seconds_nn",
        "seconds_teacher",
    ]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    table = [[fmt(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in table)) for i, c in enumerate(cols)]

    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
    print(header)
    print(sep)
    for row in table:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(cols))))


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run teacher-vs-NN ablation comparisons and summarize metrics.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sdre_control_net.pt", "sdre_control_net_P_are0.pt", "sdre_control_net_P_are1e-3.pt"],
        help="List of model .pt files to compare",
    )
    parser.add_argument("--tf", type=float, default=6000.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--target", type=str, default=None, choices=["u_net", "u_p"], help="Override target")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write CSV")
    args = parser.parse_args()

    rows = []
    for m in args.models:
        p = Path(m)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        print(f"Running: {p} (tf={args.tf}, dt={args.dt})", flush=True)
        rows.append(summarize_one(p, tf=args.tf, dt=args.dt, target_override=args.target))

    print("\nSummary:")
    print_table(rows)

    if args.csv:
        out_path = Path(args.csv)
        write_csv(rows, out_path)
        print(f"\nWrote CSV: {out_path}")


if __name__ == "__main__":
    main()
