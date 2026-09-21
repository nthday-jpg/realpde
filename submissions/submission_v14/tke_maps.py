#!/usr/bin/env python3
"""Log the SPATIAL distribution of TKE for a TTT submission (v14 companion).

Scalar TKE hides where the energy error lives. For each window this logs ::

    KE(x) = 1/2 [ Var_t(u)(x) + Var_t(v)(x) ]      # (H, W) map, Var over T

for the TTA prediction and the target, and writes the revealing panel ::

    diff(x) = KE_TTA(x) - KE_target(x)

Run::

    python submissions/submission_v14/tke_maps.py --submission submissions/submission_v14 \
        --data ./example_data --out tke_out_v14 --max-windows 4

Outputs under ``--out``: per-window ``ke_win{i:03d}.png`` (target | TTA | diff),
``ke_mean_diff.png`` (run-averaged diff), and ``ke_maps.npz`` (ke_pred, ke_tgt,
diff stacks + ids). Prints the scalar cross-check against ``scoring.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from local_eval import Normalizer, build_stream, import_submission  # noqa: E402


def ke_map(x: np.ndarray) -> np.ndarray:
    """Spatial TKE map of one window: x (T, H, W, C) -> (H, W).

    Mirrors scoring.py::kinetic_energy (variance over the T axis, u/v only).
    """
    u = x[..., 0].astype(np.float64)
    v = x[..., 1].astype(np.float64)
    var_u = ((u - u.mean(axis=0)) ** 2).mean(axis=0)
    var_v = ((v - v.mean(axis=0)) ** 2).mean(axis=0)
    return (0.5 * (var_u + var_v)).astype(np.float32)


def run_stream(submission_dir: Path, data_dir: Path, device: str):
    stats_path = data_dir / "mean_std_real.pt"
    if not stats_path.exists():
        raise SystemExit(f"Missing {stats_path}.")
    stream = build_stream(data_dir)
    normalizer = Normalizer(stats_path).to(device)
    module = import_submission(submission_dir)
    model = module.get_ttt_model(str(submission_dir), device)
    for attr in ("reset_ttt_state", "ttt_step"):
        if not hasattr(model, attr):
            raise SystemExit(f"get_ttt_model() returned an object lacking {attr}().")

    preds, tgts, ids = [], [], []
    prev_pair = None
    for step in stream:
        inp = step["input"].unsqueeze(0).to(device)
        tgt = step["target"].unsqueeze(0).to(device)
        if step["is_first"]:
            model.reset_ttt_state()
            prev_pair = None
        inp_norm, tgt_norm = normalizer.preprocess(inp, tgt)
        prev_target = prev_pair[1] if prev_pair is not None else None
        # NOTE: no torch.no_grad() here — ttt_step adapts with gradients
        # internally and predicts under its own no-grad scope.
        pred_norm, _ = model.ttt_step(inp_norm, prev_target)
        prev_pair = (inp_norm.detach(), tgt_norm.detach())
        pred_raw = normalizer.postprocess_pred(torch.as_tensor(pred_norm).detach())
        preds.append(pred_raw.squeeze(0).cpu().numpy().astype(np.float32))
        tgts.append(tgt.squeeze(0).cpu().numpy().astype(np.float32))
        ids.append(f"{step['sim_id']}@t{step['time_id']}")
    return np.stack(preds), np.stack(tgts), ids


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default=str(HERE / "submissions" / "submission_v14"))
    ap.add_argument("--data", default=str(HERE / "example_data"))
    ap.add_argument("--out", default="tke_out_v14")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-windows", type=int, default=4,
                    help="per-window panels to write (0 = all)")
    args = ap.parse_args()

    submission_dir = Path(args.submission).resolve()
    data_dir = Path(args.data).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pred_all, tgt_all, ids = run_stream(submission_dir, data_dir, args.device)
    ke_pred = np.stack([ke_map(w) for w in pred_all])  # (N, H, W)
    ke_tgt = np.stack([ke_map(w) for w in tgt_all])
    diff = ke_pred - ke_tgt

    np.savez_compressed(out / "ke_maps.npz", ke_pred=ke_pred, ke_tgt=ke_tgt,
                        diff=diff, ids=np.array(ids))

    # Scalar cross-check: mean rel-L2 between KE maps == scoring tke error.
    denom = np.linalg.norm(ke_tgt.reshape(len(ke_tgt), -1), axis=1).clip(min=1e-8)
    tke_scalar = float(np.mean(
        np.linalg.norm((ke_pred - ke_tgt).reshape(len(diff), -1), axis=1) / denom))
    try:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import scoring as official
        from local_eval import measured_channels
        c = measured_channels(tgt_all)
        ref = float(np.mean(official.tke_rel_l2_per_sample(pred_all, tgt_all, c)))
        print(f"[tke_maps] scalar TKE rel-L2 (maps): {tke_scalar:.6f} | scoring.py: {ref:.6f}")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"[tke_maps] scalar TKE rel-L2 (maps): {tke_scalar:.6f} "
              f"(scoring cross-check skipped: {exc})")
    print(f"[tke_maps] mean |diff|: {np.mean(np.abs(diff)):.6e} | "
          f"diff range [{diff.min():.3e}, {diff.max():.3e}]")

    n = len(diff) if not args.max_windows or args.max_windows <= 0 \
        else min(len(diff), args.max_windows)
    vmax_t = float(max(ke_tgt[:n].max(), ke_pred[:n].max(), 1e-12))
    for i in range(n):
        vmax_d = float(np.abs(diff[i]).max())
        if not np.isfinite(vmax_d) or vmax_d == 0.0:
            vmax_d = 1e-12
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
        im0 = ax[0].imshow(ke_tgt[i], vmin=0, vmax=vmax_t)
        ax[0].set_title(f"KE_target  {ids[i]}")
        im1 = ax[1].imshow(ke_pred[i], vmin=0, vmax=vmax_t)
        ax[1].set_title("KE_TTA")
        im2 = ax[2].imshow(diff[i], cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
        ax[2].set_title("KE_TTA - KE_target")
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        fig.suptitle(f"window {i}  scalar|diff|_rel={np.linalg.norm(diff[i]) / max(np.linalg.norm(ke_tgt[i]), 1e-8):.4f}")
        fig.tight_layout()
        fig.savefig(out / f"ke_win{i:03d}.png", dpi=120)
        plt.close(fig)

    mean_diff = diff.mean(axis=0)
    vmax_m = float(np.abs(mean_diff).max()) or 1e-12
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mean_diff, cmap="RdBu_r", vmin=-vmax_m, vmax=vmax_m)
    ax.set_title(f"mean KE_TTA - KE_target  ({len(diff)} windows)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "ke_mean_diff.png", dpi=120)
    plt.close(fig)
    print(f"[tke_maps] wrote {n} window panels + ke_mean_diff.png + ke_maps.npz to {out}")


if __name__ == "__main__":
    main()
