import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from python_SDRE import SpacecraftGame


@dataclass
class ValuePINNConfig:
    a_c_km: float = 15000.0
    e_c: float = 0.5
    gamma: float = float(np.sqrt(2.0))

    n_samples: int = 8000
    seed: int = 0

    # sampling ranges (LVLH state)
    pos_range_km: float = 1000.0
    vel_range_km_s: float = 0.05

    # training
    batch_size: int = 512
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-6

    hidden: int = 256
    layers: int = 3

    # loss weights
    lambda_bc: float = 1.0

    # residual normalization
    norm_mode: str = "residual"  # none | residual | terms
    w_residual: float = 1.0
    w_xQx: float = 0.0
    w_up: float = 0.0
    w_ue: float = 0.0
    w_adv: float = 0.0

    # whether the implied policy is u_net or u_p
    target: str = "u_net"  # u_net | u_p


class MLPValue(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")

        blocks = []
        last = in_dim
        for _ in range(layers):
            blocks.append(nn.Linear(last, hidden))
            blocks.append(nn.Tanh())
            last = hidden
        blocks.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def sample_states(cfg: ValuePINNConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    pos = rng.uniform(-cfg.pos_range_km, cfg.pos_range_km, size=(n, 3))
    vel = rng.uniform(-cfg.vel_range_km_s, cfg.vel_range_km_s, size=(n, 3))
    return np.hstack([pos, vel]).astype(np.float32)


def build_dataset(cfg: ValuePINNConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    X = sample_states(cfg, rng, cfg.n_samples)
    return X


def dynamics_f(game: SpacecraftGame, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Relative LVLH dynamics used in python_SDRE.py, implemented in torch.

    x: (B,6) [km, km/s]
    u: (B,3) [km/s^2]
    returns f: (B,6)
    """
    mu = torch.tensor(398600.4418, dtype=torch.float32, device=x.device)
    Rc = torch.tensor(float(game.Rc), dtype=torch.float32, device=x.device)
    n = torch.tensor(float(game.n), dtype=torch.float32, device=x.device)

    px, py, pz, vx, vy, vz = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

    r_d = torch.sqrt((Rc + px) ** 2 + py**2 + pz**2)

    ax0 = 2.0 * n * vy + (n**2) * px + mu / (Rc**2) - mu * (Rc + px) / (r_d**3)
    ay0 = -2.0 * n * vx + (n**2) * py - mu * py / (r_d**3)
    az0 = -mu * pz / (r_d**3)

    ax = ax0 + u[:, 0]
    ay = ay0 + u[:, 1]
    az = az0 + u[:, 2]

    f = torch.stack([vx, vy, vz, ax, ay, az], dim=1)
    return f


def hji_residual(game: SpacecraftGame, x_raw: torch.Tensor, x_norm: torch.Tensor, V: torch.Tensor, x_std: torch.Tensor, target: str) -> torch.Tensor:
    """Stationary HJI/HJB residual for pursuit-evasion LQ game:

    0 = x^T Q x + u_p^T Rp u_p - u_e^T Re u_e + gradV^T f(x, u_p + u_e)

    where
      u_p* = -0.5 inv(Rp) B^T gradV
      u_e* = +0.5 inv(Re) B^T gradV
    and B^T gradV selects the last 3 components (velocity-partial).
    """
    # grad wrt normalized input, then chain-rule to physical x
    gradV_norm = torch.autograd.grad(V.sum(), x_norm, create_graph=True)[0]
    gradV = gradV_norm / x_std  # (B,6)

    grad_v = gradV[:, 3:6]  # (B,3)

    inv_Rp = torch.tensor(game.inv_Rp.astype(np.float32), device=x_raw.device)
    inv_Re = torch.tensor(game.inv_Re.astype(np.float32), device=x_raw.device)
    Rp = torch.tensor(game.Rp.astype(np.float32), device=x_raw.device)
    Re = torch.tensor(game.Re.astype(np.float32), device=x_raw.device)
    Q = torch.tensor(game.Q.astype(np.float32), device=x_raw.device)

    u_p = -0.5 * (grad_v @ inv_Rp.T)
    u_e = +0.5 * (grad_v @ inv_Re.T)

    if target == "u_p":
        u = u_p
    elif target == "u_net":
        u = u_p + u_e
    else:
        raise ValueError(f"Unknown target: {target}")

    # costs
    xQx = torch.einsum("bi,ij,bj->b", x_raw, Q, x_raw)
    up = torch.einsum("bi,ij,bj->b", u_p, Rp, u_p)
    ue = torch.einsum("bi,ij,bj->b", u_e, Re, u_e)

    f = dynamics_f(game, x_raw, u)
    adv = torch.sum(gradV * f, dim=1)

    # HJI residual
    res = xQx + up - ue + adv
    return res


def approx_scales_from_P0(game: SpacecraftGame, X: np.ndarray, target: str) -> dict:
    """Compute fixed reference scales for normalizing HJI terms.

    Uses a constant Riccati solution P0 at the origin (A(x=0)) to approximate gradV and u.
    This is fast and gives stable, model-independent scales.
    """
    x0 = np.zeros(6, dtype=float)
    A0, Bp0, Be0 = game.get_sdc_matrices(x0)
    P0 = game.solve_game_riccati(A0, Bp0, Be0)
    if P0 is None:
        raise RuntimeError("Failed to compute P0 at origin; check game parameters")

    Q = game.Q
    Rp = game.Rp
    Re = game.Re

    # V(x) ~ x^T P0 x => gradV = 2 P0 x, so grad_v = 2(P0 x)[3:6]
    gradV = (2.0 * (X @ P0.T)).astype(np.float64)  # (N,6)
    grad_v = gradV[:, 3:6]

    u_p = -0.5 * (grad_v @ game.inv_Rp.T)
    u_e = +0.5 * (grad_v @ game.inv_Re.T)
    u = u_p if target == "u_p" else (u_p + u_e)

    xQx = np.einsum("bi,ij,bj->b", X, Q, X)
    up = np.einsum("bi,ij,bj->b", u_p, Rp, u_p)
    ue = np.einsum("bi,ij,bj->b", u_e, Re, u_e)

    # dynamics term using the same model as training
    mu = 398600.4418
    Rc = float(game.Rc)
    n = float(game.n)
    px, py, pz, vx, vy, vz = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    r_d = np.sqrt((Rc + px) ** 2 + py**2 + pz**2)
    ax0 = 2.0 * n * vy + (n**2) * px + mu / (Rc**2) - mu * (Rc + px) / (r_d**3)
    ay0 = -2.0 * n * vx + (n**2) * py - mu * py / (r_d**3)
    az0 = -mu * pz / (r_d**3)
    f = np.stack([vx, vy, vz, ax0 + u[:, 0], ay0 + u[:, 1], az0 + u[:, 2]], axis=1)
    adv = np.sum(gradV * f, axis=1)

    res = xQx + up - ue + adv

    def med_abs(a: np.ndarray) -> float:
        return float(np.median(np.abs(a)))

    eps = 1e-12
    scales = {
        "s_xQx": med_abs(xQx) + eps,
        "s_up": med_abs(up) + eps,
        "s_ue": med_abs(ue) + eps,
        "s_adv": med_abs(adv) + eps,
        "s_res": med_abs(res) + eps,
    }
    return scales


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a value-function PINN (stationary HJI residual) for the circular LVLH SDRE game.")
    parser.add_argument("--out", type=str, default="models/value/sdre_value_net.pt", help="Output model path")
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pos-range-km", type=float, default=1000.0)
    parser.add_argument("--vel-range-km-s", type=float, default=0.05)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--target", type=str, default="u_net", choices=["u_net", "u_p"])
    parser.add_argument("--norm-mode", type=str, default="residual", choices=["none", "residual", "terms"], help="How to normalize HJI loss")
    parser.add_argument("--w-residual", type=float, default=1.0)
    parser.add_argument("--w-xQx", type=float, default=0.0)
    parser.add_argument("--w-up", type=float, default=0.0)
    parser.add_argument("--w-ue", type=float, default=0.0)
    parser.add_argument("--w-adv", type=float, default=0.0)
    args = parser.parse_args()

    cfg = ValuePINNConfig(
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pos_range_km=args.pos_range_km,
        vel_range_km_s=args.vel_range_km_s,
        lambda_bc=float(args.lambda_bc),
        target=args.target,
        norm_mode=str(args.norm_mode),
        w_residual=float(args.w_residual),
        w_xQx=float(args.w_xQx),
        w_up=float(args.w_up),
        w_ue=float(args.w_ue),
        w_adv=float(args.w_adv),
    )

    game = SpacecraftGame(chief_semi_major_axis=cfg.a_c_km, chief_eccentricity=cfg.e_c, gamma=cfg.gamma)

    X = build_dataset(cfg)  # (N,6)

    ref_scales = approx_scales_from_P0(game, X.astype(np.float64), target=cfg.target)
    print(
        "Reference scales (median abs, from P0 approximation): "
        + ", ".join([f"{k}={v:.6g}" for k, v in ref_scales.items()])
    )

    # normalization for stable training
    x_mean = X.mean(axis=0)
    x_std_np = X.std(axis=0) + 1e-8
    Xn = (X - x_mean) / x_std_np

    # tensors
    device = torch.device("cpu")
    X_raw = torch.from_numpy(X).to(device)
    X_norm = torch.from_numpy(Xn).to(device)

    ds = TensorDataset(X_norm, X_raw)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = MLPValue(in_dim=6, hidden=cfg.hidden, layers=cfg.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_std = torch.tensor(x_std_np, dtype=torch.float32, device=device)
    s_xQx = torch.tensor(ref_scales["s_xQx"], dtype=torch.float32, device=device)
    s_up = torch.tensor(ref_scales["s_up"], dtype=torch.float32, device=device)
    s_ue = torch.tensor(ref_scales["s_ue"], dtype=torch.float32, device=device)
    s_adv = torch.tensor(ref_scales["s_adv"], dtype=torch.float32, device=device)
    s_res = torch.tensor(ref_scales["s_res"], dtype=torch.float32, device=device)

    best = float("inf")
    best_state = None

    x0_norm = torch.zeros((1, 6), dtype=torch.float32, device=device)
    x0_raw = torch.zeros((1, 6), dtype=torch.float32, device=device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n_seen = 0

        for xb_norm, xb_raw in dl:
            xb_norm = xb_norm.to(device)
            xb_raw = xb_raw.to(device)

            xb_norm = xb_norm.clone().detach().requires_grad_(True)
            Vb = model(xb_norm)

            # Compute terms for optional component-wise normalization
            gradV_norm = torch.autograd.grad(Vb.sum(), xb_norm, create_graph=True)[0]
            gradV = gradV_norm / x_std
            grad_v = gradV[:, 3:6]

            inv_Rp = torch.tensor(game.inv_Rp.astype(np.float32), device=device)
            inv_Re = torch.tensor(game.inv_Re.astype(np.float32), device=device)
            Rp = torch.tensor(game.Rp.astype(np.float32), device=device)
            Re = torch.tensor(game.Re.astype(np.float32), device=device)
            Q = torch.tensor(game.Q.astype(np.float32), device=device)

            u_p = -0.5 * (grad_v @ inv_Rp.T)
            u_e = +0.5 * (grad_v @ inv_Re.T)
            u = u_p if cfg.target == "u_p" else (u_p + u_e)

            xQx = torch.einsum("bi,ij,bj->b", xb_raw, Q, xb_raw)
            up = torch.einsum("bi,ij,bj->b", u_p, Rp, u_p)
            ue = torch.einsum("bi,ij,bj->b", u_e, Re, u_e)

            f = dynamics_f(game, xb_raw, u)
            adv = torch.sum(gradV * f, dim=1)

            res = xQx + up - ue + adv

            if cfg.norm_mode == "none":
                loss_pde = torch.mean(res * res)
            elif cfg.norm_mode == "residual":
                loss_pde = torch.mean((res / s_res) ** 2)
            elif cfg.norm_mode == "terms":
                # Sum of normalized term energies, plus (optionally) residual energy.
                loss_pde = torch.tensor(0.0, device=device)
                if cfg.w_xQx != 0.0:
                    loss_pde = loss_pde + float(cfg.w_xQx) * torch.mean((xQx / s_xQx) ** 2)
                if cfg.w_up != 0.0:
                    loss_pde = loss_pde + float(cfg.w_up) * torch.mean((up / s_up) ** 2)
                if cfg.w_ue != 0.0:
                    loss_pde = loss_pde + float(cfg.w_ue) * torch.mean((ue / s_ue) ** 2)
                if cfg.w_adv != 0.0:
                    loss_pde = loss_pde + float(cfg.w_adv) * torch.mean((adv / s_adv) ** 2)
                if cfg.w_residual != 0.0:
                    loss_pde = loss_pde + float(cfg.w_residual) * torch.mean((res / s_res) ** 2)
            else:
                raise ValueError(f"Unknown norm_mode: {cfg.norm_mode}")

            # boundary condition at origin: V(0)=0
            x0_norm_req = x0_norm.clone().detach().requires_grad_(True)
            V0 = model(x0_norm_req)
            loss_bc = torch.mean(V0 * V0)

            loss = loss_pde + float(cfg.lambda_bc) * loss_bc

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item()) * xb_norm.shape[0]
            n_seen += xb_norm.shape[0]

        train_loss = total / max(1, n_seen)

        # simple "best" tracking on training loss (no labels)
        if train_loss < best:
            best = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(f"Epoch {epoch:03d}: loss={train_loss:.6g} best={best:.6g} (mode={cfg.norm_mode})")

    if best_state is None:
        best_state = model.state_dict()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "model": {
            "in_dim": 6,
            "out_dim": 1,
            "hidden": int(cfg.hidden),
            "layers": int(cfg.layers),
            "activation": "tanh",
            "type": "V",
        },
        "norm": {
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std_np.astype(np.float32),
            "y_mean": np.zeros((1,), dtype=np.float32),
            "y_std": np.ones((1,), dtype=np.float32),
        },
        "scales": {k: float(v) for k, v in ref_scales.items()},
        "state_dict": best_state,
    }

    torch.save(payload, out_path)
    print(f"Saved model to {out_path}")

    meta = {
        "config": asdict(cfg),
        "model": payload["model"],
        "scales": payload["scales"],
        "best_train_loss": float(best),
    }
    out_meta = out_path.with_suffix(".json")
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved meta to {out_meta}")


if __name__ == "__main__":
    main()
