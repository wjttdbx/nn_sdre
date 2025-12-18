import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from python_SDRE import SpacecraftGame
from python_SDRE_elliptic import (
    SpacecraftGameElliptic,
    chief_perigee_state,
    inertial_from_rel,
    rel_from_inertial,
    two_body_accel,
)


@dataclass
class TrainConfig:
    a_c_km: float = 15000.0
    e_c: float = 0.5
    gamma: float = float(np.sqrt(2.0))

    n_samples: int = 8000
    seed: int = 0

    # sampling ranges (LVLH state)
    pos_range_km: float = 1000.0
    vel_range_km_s: float = 0.05

    # dataset generation
    data_mode: str = "trajectory"  # trajectory | uniform
    teacher: str = "elliptic"  # elliptic | circular
    tf_s: float = 2000.0
    dt_s: float = 10.0
    n_trajectories: int = 32
    init_center_pos_km: float = 500.0
    init_center_vel_km_s: float = 0.01

    # training
    batch_size: int = 512
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-6

    hidden: int = 256
    layers: int = 3

    model_type: str = "u"  # u | P
    lambda_are: float = 0.0

    # output choice
    target: str = "u_net"  # u_net | u_p


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")

        blocks = []
        last = in_dim
        for _ in range(layers):
            blocks.append(nn.Linear(last, hidden))
            blocks.append(nn.Tanh())
            last = hidden
        blocks.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def teacher_controls(game: SpacecraftGame, x: np.ndarray, target: str) -> np.ndarray | None:
    # Use the existing SDRE solver as the teacher.
    A, Bp, Be = game.get_sdc_matrices(x)
    P = game.solve_game_riccati(A, Bp, Be)
    if P is None:
        return None

    Px = P @ x
    BTPx = Px[3:6]
    u_p = -game.inv_Rp @ BTPx
    u_e = game.inv_Re @ BTPx

    if target == "u_p":
        return u_p
    if target == "u_net":
        return u_p + u_e
    raise ValueError(f"Unknown target: {target}")


def sample_states(cfg: TrainConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    pos = rng.uniform(-cfg.pos_range_km, cfg.pos_range_km, size=(n, 3))
    vel = rng.uniform(-cfg.vel_range_km_s, cfg.vel_range_km_s, size=(n, 3))
    return np.hstack([pos, vel]).astype(np.float64)


def _rk4_step(fun, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = fun(t, y)
    k2 = fun(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = fun(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = fun(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _sample_initial_rel(cfg: TrainConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample an initial LVLH relative state for trajectory rollouts.

    Centered around the Table-1 nominal values with uniform jitter.
    """
    center_rho = np.array([cfg.init_center_pos_km] * 3, dtype=float)
    center_rhodot = np.array([cfg.init_center_vel_km_s] * 3, dtype=float)

    rho0 = center_rho + rng.uniform(-cfg.pos_range_km, cfg.pos_range_km, size=(3,))
    rhodot0 = center_rhodot + rng.uniform(-cfg.vel_range_km_s, cfg.vel_range_km_s, size=(3,))
    return rho0, rhodot0


def _elliptic_rollout_pairs(cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (x_rel, u_teacher) by rolling out the accurate elliptic teacher."""
    if cfg.model_type != "u":
        raise ValueError(
            "trajectory+elliptic teacher currently supports only model-type=u. "
            "The elliptic teacher is time-varying (depends on orbital phase), so learning a pure P(x) mapping is ill-posed."
        )

    rng = np.random.default_rng(cfg.seed)
    game = SpacecraftGameElliptic(a_c=cfg.a_c_km, e_c=cfg.e_c, gamma=cfg.gamma)

    # chief initial state (perigee)
    r_c0, v_c0 = chief_perigee_state(cfg.a_c_km, cfg.e_c)

    n_steps = int(np.floor(cfg.tf_s / cfg.dt_s))
    if n_steps < 1:
        raise ValueError("tf_s must be >= dt_s")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    def f_dyn(t: float, y: np.ndarray) -> np.ndarray:
        chief_r = y[0:3]
        chief_v = y[3:6]
        deputy_r = y[6:9]
        deputy_v = y[9:12]

        rho, rho_dot, C_LI, _, _ = rel_from_inertial(chief_r, chief_v, deputy_r, deputy_v)
        x_rel = np.hstack((rho, rho_dot))

        A = game.sdc_A(chief_r, chief_v, x_rel)
        u_p, u_e = game.get_control(A, x_rel)
        u_net_l = u_p + u_e

        a_c = two_body_accel(chief_r)
        a_d = two_body_accel(deputy_r) + (C_LI @ u_net_l)
        return np.hstack((chief_v, a_c, deputy_v, a_d)).astype(float)

    traj_attempts = 0
    while len(xs) < cfg.n_samples:
        traj_attempts += 1
        rho0, rhodot0 = _sample_initial_rel(cfg, rng)
        r_d0, v_d0 = inertial_from_rel(r_c0, v_c0, rho0, rhodot0)
        y = np.hstack((r_c0, v_c0, r_d0, v_d0)).astype(float)
        t = 0.0

        for _ in range(n_steps + 1):
            chief_r = y[0:3]
            chief_v = y[3:6]
            deputy_r = y[6:9]
            deputy_v = y[9:12]
            rho, rho_dot, _, _, _ = rel_from_inertial(chief_r, chief_v, deputy_r, deputy_v)
            x_rel = np.hstack((rho, rho_dot))

            A = game.sdc_A(chief_r, chief_v, x_rel)
            u_p, u_e = game.get_control(A, x_rel)
            if cfg.target == "u_p":
                u = u_p
            else:
                u = u_p + u_e

            if not (np.all(np.isfinite(x_rel)) and np.all(np.isfinite(u))):
                break

            xs.append(x_rel.astype(np.float64))
            ys.append(u.astype(np.float64))
            if len(xs) >= cfg.n_samples:
                break

            y = _rk4_step(f_dyn, t, y, float(cfg.dt_s))
            t += float(cfg.dt_s)

        if traj_attempts % 5 == 0:
            print(f"Rollout progress: {len(xs)}/{cfg.n_samples} samples collected...")

    X = np.stack(xs[: cfg.n_samples], axis=0).astype(np.float32)
    Y = np.stack(ys[: cfg.n_samples], axis=0).astype(np.float32)
    print(f"Dataset(trajectory): {X.shape[0]} samples from rollouts (dt={cfg.dt_s}, tf={cfg.tf_s})")
    return X, Y


def build_dataset(cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.data_mode == "trajectory":
        if cfg.teacher != "elliptic":
            raise ValueError("trajectory mode currently supports only teacher=elliptic")
        return _elliptic_rollout_pairs(cfg)

    if cfg.data_mode != "uniform":
        raise ValueError(f"Unknown data_mode: {cfg.data_mode}")

    if cfg.teacher != "circular":
        raise ValueError("uniform mode currently supports only teacher=circular")

    rng = np.random.default_rng(cfg.seed)
    game = SpacecraftGame(chief_semi_major_axis=cfg.a_c_km, chief_eccentricity=cfg.e_c, gamma=cfg.gamma)

    xs = []
    ys = []

    batch = 4096
    attempts = 0
    while len(xs) < cfg.n_samples:
        x_cand = sample_states(cfg, rng, batch)
        for x in x_cand:
            y = teacher_controls(game, x, cfg.target)
            attempts += 1
            if y is None:
                continue
            xs.append(x)
            ys.append(y)
            if len(xs) >= cfg.n_samples:
                break

    X = np.stack(xs, axis=0).astype(np.float32)
    Y = np.stack(ys, axis=0).astype(np.float32)
    kept_ratio = float(len(xs) / attempts)
    print(f"Dataset(uniform): {X.shape[0]} samples kept (kept_ratio={kept_ratio:.3f})")
    return X, Y


def save_dataset_cache(path: Path, X: np.ndarray, Y: np.ndarray, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": asdict(cfg),
        "n_samples": int(X.shape[0]),
        "data_mode": cfg.data_mode,
        "teacher": cfg.teacher,
    }
    np.savez(path, X=X.astype(np.float32), Y=Y.astype(np.float32), meta=json.dumps(meta, ensure_ascii=False))
    print(f"Cached dataset to {path} (X{X.shape}, Y{Y.shape})")


def load_dataset_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict | None]:
    data = np.load(path, allow_pickle=False)
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)
    meta = None
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"].item()))
        except Exception:
            meta = None
    print(f"Loaded cached dataset from {path} (X{X.shape}, Y{Y.shape})")
    return X, Y, meta


def are_residual(game: SpacecraftGame, x: np.ndarray, P: np.ndarray) -> np.ndarray:
    """GARE residual: A^T P + P A - P S P + Q, where S = B inv(Rp) B^T - B inv(Re) B^T."""
    A, Bp, Be = game.get_sdc_matrices(x)

    S = np.zeros((6, 6), dtype=np.float32)
    S[3:6, 3:6] = (game.inv_Rp - game.inv_Re).astype(np.float32)

    # residual
    R = A.T.astype(np.float32) @ P + P @ A.astype(np.float32) - P @ S @ P + game.Q.astype(np.float32)
    return R


def sym6_from_upper21(u: torch.Tensor) -> torch.Tensor:
    """Convert (B,21) upper-tri entries to symmetric (B,6,6)."""
    if u.ndim != 2 or u.shape[1] != 21:
        raise ValueError(f"Expected (B,21), got {tuple(u.shape)}")
    B = u.shape[0]
    P = torch.zeros((B, 6, 6), dtype=u.dtype, device=u.device)
    k = 0
    for i in range(6):
        for j in range(i, 6):
            P[:, i, j] = u[:, k]
            P[:, j, i] = u[:, k]
            k += 1
    return P


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a torch surrogate for SDRE control.")
    parser.add_argument("--out", type=str, default="models/control/sdre_control_net.pt", help="Output model path")
    parser.add_argument("--data-mode", type=str, default="trajectory", choices=["trajectory", "uniform"], help="How to sample states")
    parser.add_argument("--teacher", type=str, default="elliptic", choices=["elliptic", "circular"], help="Which teacher model to label data")
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument("--tf", type=float, default=2000.0, help="Trajectory rollout horizon [s] (trajectory mode)")
    parser.add_argument("--dt", type=float, default=10.0, help="Trajectory rollout step [s] (trajectory mode)")
    parser.add_argument("--n-trajectories", type=int, default=32, help="Number of rollouts to try (trajectory mode; may be exceeded to reach n-samples)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to a cached dataset (.npz). If missing, a new dataset will be generated and saved to this path.")
    parser.add_argument("--save-dataset", type=str, default=None, help="Force saving the newly generated dataset to this path (.npz). Useful when --dataset is not set.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pos-range-km", type=float, default=1000.0)
    parser.add_argument("--vel-range-km-s", type=float, default=0.05)
    parser.add_argument("--target", type=str, default="u_net", choices=["u_net", "u_p"])
    parser.add_argument("--model-type", type=str, default="u", choices=["u", "P"], help="Train network to output u (3) or symmetric P (21)")
    parser.add_argument("--lambda-are", type=float, default=0.0, help="Weight for ARE residual loss (only for model-type=P)")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_mode=args.data_mode,
        teacher=args.teacher,
        n_samples=args.n_samples,
        tf_s=float(args.tf),
        dt_s=float(args.dt),
        n_trajectories=int(args.n_trajectories),
        epochs=args.epochs,
        batch_size=args.batch_size,
        pos_range_km=args.pos_range_km,
        vel_range_km_s=args.vel_range_km_s,
        target=args.target,
        model_type=args.model_type,
        lambda_are=float(args.lambda_are),
    )

    dataset_path = Path(args.dataset) if args.dataset else None
    save_dataset_path = Path(args.save_dataset) if args.save_dataset else None
    if dataset_path is not None and save_dataset_path is None and not dataset_path.exists():
        # If user specifies a dataset path that does not exist, reuse it as the save path.
        save_dataset_path = dataset_path

    # The circular game is only needed for uniform/circular mode and the optional differentiable ARE residual term.
    circular_game = None
    if cfg.teacher == "circular" or cfg.lambda_are > 0.0:
        circular_game = SpacecraftGame(chief_semi_major_axis=cfg.a_c_km, chief_eccentricity=cfg.e_c, gamma=cfg.gamma)

    meta = None
    if dataset_path is not None and dataset_path.exists():
        X, Y_teacher, meta = load_dataset_cache(dataset_path)
        if meta and "config" in meta:
            cfg_meta = meta["config"]
            mismatches = []
            for key in ["data_mode", "teacher", "tf_s", "dt_s", "n_samples"]:
                if key in cfg_meta and getattr(cfg, key, None) != cfg_meta[key]:
                    mismatches.append((key, cfg_meta[key], getattr(cfg, key, None)))
            if mismatches:
                print("Warning: cached dataset config differs from current args:")
                for k, old, new in mismatches:
                    print(f"  {k}: cached={old} current={new}")
    else:
        X, Y_teacher = build_dataset(cfg)
        if save_dataset_path is not None:
            save_dataset_cache(save_dataset_path, X, Y_teacher, cfg)

    # normalization (always normalize inputs; outputs depend on model type)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    Xn = (X - x_mean) / x_std

    if cfg.model_type == "u":
        Y = Y_teacher
        y_mean = Y.mean(axis=0)
        y_std = Y.std(axis=0) + 1e-8
        Yn = (Y - y_mean) / y_std
        out_dim = int(Y.shape[1])
    elif cfg.model_type == "P":
        # Network outputs upper-triangular entries of P (21). We'll supervise u, and optionally add ARE residual.
        # For convenience, keep a dummy y normalization of shape (21,).
        out_dim = 21
        y_mean = np.zeros((out_dim,), dtype=np.float32)
        y_std = np.ones((out_dim,), dtype=np.float32)
        Yn = np.zeros((Xn.shape[0], out_dim), dtype=np.float32)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    # split
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(Xn.shape[0])
    n_train = int(0.9 * Xn.shape[0])
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train = torch.from_numpy(Xn[train_idx])
    Y_train = torch.from_numpy(Yn[train_idx])
    X_val = torch.from_numpy(Xn[val_idx])
    Y_val = torch.from_numpy(Yn[val_idx])

    # Keep unnormalized X for computing A(x) inside ARE residual.
    X_raw_train = torch.from_numpy(X[train_idx])
    X_raw_val = torch.from_numpy(X[val_idx])
    Y_teacher_train = torch.from_numpy(Y_teacher[train_idx])
    Y_teacher_val = torch.from_numpy(Y_teacher[val_idx])

    ds_train = TensorDataset(X_train, Y_train, X_raw_train, Y_teacher_train)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = MLP(in_dim=6, out_dim=out_dim, hidden=cfg.hidden, layers=cfg.layers)
    device = torch.device("cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb, x_raw_b, y_teacher_b in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            x_raw_b = x_raw_b.to(device)
            y_teacher_b = y_teacher_b.to(device)

            pred = model(xb)

            if cfg.model_type == "u":
                loss = loss_fn(pred, yb)
            else:
                # model_type == "P": supervise u derived from P, plus optional ARE residual.
                P = sym6_from_upper21(pred)
                # u_p = -inv(Rp) * (P x)_vel
                Px = torch.bmm(P, x_raw_b.unsqueeze(-1)).squeeze(-1)
                BTPx = Px[:, 3:6]
                # Table-1 weights: Rp = 1e13 I, Re = gamma^2 Rp
                inv_rp_scalar = torch.tensor(1e-13, dtype=torch.float32, device=device)
                inv_Rp = inv_rp_scalar * torch.eye(3, dtype=torch.float32, device=device)
                inv_Re = (inv_rp_scalar / (float(cfg.gamma) ** 2)) * torch.eye(3, dtype=torch.float32, device=device)
                u_p = -(BTPx @ inv_Rp.T)
                u_e = (BTPx @ inv_Re.T)
                u_net = u_p + u_e
                u_pred = u_p if cfg.target == "u_p" else u_net
                loss_u = loss_fn(u_pred, y_teacher_b)

                loss_are = torch.tensor(0.0, device=device)
                if cfg.lambda_are > 0.0:
                    if circular_game is None:
                        raise RuntimeError("Internal error: circular teacher is required for lambda_are > 0")
                    # Compute ARE residual in numpy (per-sample) then backprop through P is not possible.
                    # So we compute a differentiable surrogate residual using torch with A(x) approximated via the teacher function is not available.
                    # To keep it simple and correct, we implement A(x) in torch using the same formulas as python_SDRE.get_sdc_matrices.
                    # (This keeps gradients w.r.t P, enabling physics-informed regularization.)

                    mu = torch.tensor(398600.4418, device=device, dtype=torch.float32)
                    Rc = torch.tensor(float(circular_game.Rc), device=device, dtype=torch.float32)
                    n = torch.tensor(float(circular_game.n), device=device, dtype=torch.float32)
                    x = x_raw_b[:, 0]
                    y = x_raw_b[:, 1]
                    z = x_raw_b[:, 2]
                    Rd_sq = (Rc + x) ** 2 + y**2 + z**2
                    Rd = torch.sqrt(Rd_sq)

                    dRd_dx = (Rc + x) / Rd
                    dRd_dy = y / Rd
                    dRd_dz = z / Rd

                    term1 = -mu / Rd**3
                    term2 = 3.0 * mu / Rd**4
                    n_sq = n**2

                    A = torch.zeros((xb.shape[0], 6, 6), dtype=torch.float32, device=device)
                    A[:, 0:3, 3:6] = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
                    A[:, 3, 4] = 2.0 * n
                    A[:, 4, 3] = -2.0 * n

                    A[:, 3, 0] = n_sq + term1 + term2 * dRd_dx * (Rc + x)
                    A[:, 3, 1] = term2 * dRd_dy * (Rc + x)
                    A[:, 3, 2] = term2 * dRd_dz * (Rc + x)

                    A[:, 4, 0] = term2 * dRd_dx * y
                    A[:, 4, 1] = n_sq + term1 + term2 * dRd_dy * y
                    A[:, 4, 2] = term2 * dRd_dz * y

                    A[:, 5, 0] = term2 * dRd_dx * z
                    A[:, 5, 1] = term2 * dRd_dy * z
                    A[:, 5, 2] = term1 + term2 * dRd_dz * z

                    S = torch.zeros((xb.shape[0], 6, 6), dtype=torch.float32, device=device)
                    S[:, 3:6, 3:6] = (inv_Rp - inv_Re).unsqueeze(0)

                    Q = torch.tensor(circular_game.Q.astype(np.float32), device=device).unsqueeze(0)
                    Rmat = torch.bmm(A.transpose(1, 2), P) + torch.bmm(P, A) - torch.bmm(torch.bmm(P, S), P) + Q
                    loss_are = torch.mean(Rmat * Rmat)

                loss = loss_u + float(cfg.lambda_are) * loss_are

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            n_seen += xb.shape[0]

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            if cfg.model_type == "u":
                val_loss = float(loss_fn(val_pred, Y_val.to(device)).item())
            else:
                # report validation on supervised u only (in physical units)
                P = sym6_from_upper21(val_pred)
                Px = torch.bmm(P, X_raw_val.to(device).unsqueeze(-1)).squeeze(-1)
                BTPx = Px[:, 3:6]
                inv_rp_scalar = torch.tensor(1e-13, dtype=torch.float32, device=device)
                inv_Rp = inv_rp_scalar * torch.eye(3, dtype=torch.float32, device=device)
                inv_Re = (inv_rp_scalar / (float(cfg.gamma) ** 2)) * torch.eye(3, dtype=torch.float32, device=device)
                u_p = -(BTPx @ inv_Rp.T)
                u_e = (BTPx @ inv_Re.T)
                u_net = u_p + u_e
                u_pred = u_p if cfg.target == "u_p" else u_net
                val_loss = float(loss_fn(u_pred, Y_teacher_val.to(device)).item())

        train_loss = total / max(1, n_seen)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(f"Epoch {epoch:03d}: train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}")

    if best_state is None:
        best_state = model.state_dict()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "model": {
            "in_dim": 6,
            "out_dim": int(out_dim),
            "hidden": int(cfg.hidden),
            "layers": int(cfg.layers),
            "activation": "tanh",
            "type": cfg.model_type,
        },
        "norm": {
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        "state_dict": best_state,
    }

    torch.save(payload, out_path)
    print(f"Saved model to {out_path}")

    # also dump a small json for quick inspection (without weights)
    meta = {
        "config": asdict(cfg),
        "model": payload["model"],
        "best_val_mse_norm": best_val,
    }
    out_meta = out_path.with_suffix(".json")
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved meta to {out_meta}")


if __name__ == "__main__":
    main()
