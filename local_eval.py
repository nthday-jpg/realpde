#!/usr/bin/env python3
"""Local CPU smoke test for a RealPDE Track 2 LTTTA submission.

Self-contained: it does NOT need the downloaded ingestion/scoring programs. It
mirrors the streaming loop of the official ingestion program on the tiny
``example_data`` bundled here -- reset at trajectory boundaries, prev-target
passing, and per-step timing -- then prints the mean per-step time and a mock
relative-L2 over the measured (u, v) channels.

This is only a shape/plumbing check. The mock rel-L2 is computed on the two
synthetic example trajectories and is NOT comparable to the leaderboard; the
official ingestion.py / scoring.py are authoritative.

Usage:
    python local_eval.py --submission <dir with submission.py>
    python local_eval.py --submission ../solutions/baseline_solution
    python local_eval.py            # defaults --submission to this kit dir

The data layout mirrors CompetitionAirfoil(mode="test", dataset_type="real"):
    <data>/test_real/*.h5      (flat u, v[, p] datasets, native 64x128)
    <data>/mean_std_real.pt    (mean_inputs, mean_targets, std_inputs, std_targets)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# Fixed streaming protocol constants (must match the official pipeline).
IN_STEP = 20
OUT_STEP = 20
INTERVAL = 20
SUB_S = 2  # spatial subsample: native 64x128 -> 32x64
HORIZON = IN_STEP + OUT_STEP


# --------------------------------------------------------------------------- #
# Data pipeline (faithful minimal copy of CompetitionAirfoil + TTTDataset).
# --------------------------------------------------------------------------- #
def build_stream(data_dir: Path):
    """Return an ordered list of steps: dicts with input/target tensors + meta.

    Sorted by (filename, time_id); ``is_first`` marks trajectory boundaries.
    """
    test_dir = data_dir / "test_real"
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".h5"))
    if not files:
        raise SystemExit(f"No .h5 files under {test_dir}")

    entries = []  # (filename, time_id)
    for name in files:
        with h5py.File(test_dir / name, "r") as f:
            n_frames = f["u"].shape[0]
        max_time_id = n_frames - HORIZON
        if max_time_id < 0:
            continue
        for time_id in range(0, max_time_id + 1, INTERVAL):
            entries.append((name, time_id))
    entries.sort(key=lambda e: (e[0], e[1]))

    stream = []
    prev_name = None
    for name, time_id in entries:
        with h5py.File(test_dir / name, "r") as f:
            n_frames = f["u"].shape[0]
            end = min(time_id + HORIZON, n_frames)
            u = f["u"][time_id:end, ::SUB_S, ::SUB_S]
            v = f["v"][time_id:end, ::SUB_S, ::SUB_S]
            p = np.zeros_like(u)  # real data: p is zero-filled
        data = np.stack([u, v, p], axis=-1)
        if data.shape[0] < HORIZON:  # pad short tails by repeating last frame
            pad = np.repeat(data[-1:], HORIZON - data.shape[0], axis=0)
            data = np.concatenate([data, pad], axis=0)
        inp = torch.tensor(data[:IN_STEP], dtype=torch.float32)
        tgt = torch.tensor(data[IN_STEP:HORIZON], dtype=torch.float32)
        stream.append({
            "input": inp, "target": tgt,
            "sim_id": name, "time_id": time_id,
            "is_first": name != prev_name,
        })
        prev_name = name
    return stream


class Normalizer:
    """Minimal GaussianNormalizer: affine per-channel, std==0 -> 1."""

    def __init__(self, stats_path: Path):
        mi, mt, si, st = torch.load(stats_path, map_location="cpu", weights_only=False)
        one = torch.ones_like
        self.mean_in, self.mean_tgt = mi.float(), mt.float()
        self.std_in = torch.where(si == 0, one(si), si).float()
        self.std_tgt = torch.where(st == 0, one(st), st).float()

    def to(self, device) -> "Normalizer":
        """Move stats to the eval device (same device the model predicts on)."""
        self.mean_in = self.mean_in.to(device)
        self.mean_tgt = self.mean_tgt.to(device)
        self.std_in = self.std_in.to(device)
        self.std_tgt = self.std_tgt.to(device)
        return self

    def preprocess(self, x, y):
        c1, c2 = x.shape[-1], y.shape[-1]
        xn = (x - self.mean_in[..., :c1]) / self.std_in[..., :c1]
        yn = (y - self.mean_tgt[..., :c2]) / self.std_tgt[..., :c2]
        return xn, yn

    def postprocess_pred(self, pred_norm):
        c = pred_norm.shape[-1]
        return pred_norm * self.std_tgt[..., :c] + self.mean_tgt[..., :c]


# --------------------------------------------------------------------------- #
# Submission loading + streaming loop.
# --------------------------------------------------------------------------- #
def import_submission(submission_dir: Path):
    sub_file = submission_dir / "submission.py"
    if not sub_file.exists():
        raise SystemExit(f"Expected submission.py at {sub_file}")
    sys.path.insert(0, str(submission_dir))  # so relative imports resolve
    spec = importlib.util.spec_from_file_location("participant_submission", sub_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_ttt_model"):
        raise SystemExit("submission.py must define get_ttt_model(submission_dir, device).")
    return module


def measured_channels(target: np.ndarray) -> int:
    active = sum(not np.allclose(target[..., i], 0.0) for i in range(target.shape[-1]))
    return max(1, int(active))


def rel_l2(pred: np.ndarray, target: np.ndarray, c: int) -> float:
    p = pred[..., :c].reshape(pred.shape[0], -1)
    t = target[..., :c].reshape(target.shape[0], -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return float(np.mean(np.linalg.norm(p - t, axis=1) / denom))


# --------------------------------------------------------------------------- #
# SPS component breakdown (single source of truth for the notebook).
# --------------------------------------------------------------------------- #
# Must mirror scoring.py::aggregate_sps defaults (scorer default band,
# normalize_factor, dm/tke/mvpe weights). The *scored* (sps_raw, coverage)
# always comes from official.aggregate_sps itself so logging can never drift
# from the leaderboard; the branch/nil values below reuse the same official
# per-sample helpers purely for diagnosis.
SPS_NORMALIZE_FACTOR = 0.5
SPS_WEIGHT_DM = 0.5
SPS_WEIGHT_TKE = 0.3
SPS_WEIGHT_MVPE = 0.2
SPS_DEFAULT_HALF_FRAC = 0.05  # scorer default band: pred +/- 0.05*|pred|


def _official_scoring():
    """Import the bundled scoring.py (authoritative SPS formulas)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import scoring as official
    return official


def resolve_sps_bands(pred_all: np.ndarray, lo_raw, hi_raw):
    """Resolve per-step bounds to full stacks with scorer-default fallback.

    Steps without returned bounds get the scorer's default +/-5% band, same
    as the official evaluator. Returns (lower, upper, n_calibrated).
    """
    lo_all, hi_all = [], []
    n_cal = 0
    for step_pred, lob, hib in zip(pred_all, lo_raw, hi_raw):
        if lob is not None and hib is not None:
            lo_all.append(lob)
            hi_all.append(hib)
            n_cal += 1
        else:
            band = SPS_DEFAULT_HALF_FRAC * np.abs(step_pred)
            lo_all.append(step_pred - band)
            hi_all.append(step_pred + band)
    return (np.stack(lo_all, axis=0) if lo_all else None,
            np.stack(hi_all, axis=0) if hi_all else None, n_cal)


def sps_component_breakdown(
    pred_all: np.ndarray,
    tgt_all: np.ndarray,
    c: int,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict:
    """Decompose the official SPS score into loggable components.

    Returns dict with: sps_raw (weighted, from official.aggregate_sps),
    coverage, mean_nil, mean_exp_neg_nil, sps_dm, sps_tke, sps_mvpe, plus
    the weights/normalize factor used. ``lower``/``upper`` are resolved
    (denormalized, full-stack) bounds; when None the scorer default band
    is used, exactly as official.aggregate_sps does.
    """
    official = _official_scoring()
    sps_raw, coverage = official.aggregate_sps(
        pred_all, tgt_all, c, lower=lower, upper=upper)

    p = pred_all[..., :c]
    t = tgt_all[..., :c]
    if lower is None or upper is None:
        interval = 0.1 * np.abs(p)
        lo = p - interval / 2.0
        hi = p + interval / 2.0
    else:
        lo = np.asarray(lower, dtype=np.float32)[..., :c]
        hi = np.asarray(upper, dtype=np.float32)[..., :c]
    inside = (t >= lo) & (t <= hi)
    nil = (hi - lo) / official.SIGMA_GLOBAL

    dm = official.rel_l2_per_sample(pred_all, tgt_all, c)
    tke = official.tke_rel_l2_per_sample(pred_all, tgt_all, c)
    mvpe = official.mvpe_rel_l2_per_sample(pred_all, tgt_all)
    dm_n = dm / (SPS_NORMALIZE_FACTOR + dm)
    tke_n = tke / (SPS_NORMALIZE_FACTOR + tke)
    mvpe_n = mvpe / (SPS_NORMALIZE_FACTOR + mvpe)

    def branch(pm: np.ndarray) -> float:
        shaped = pm.reshape((pm.shape[0],) + (1,) * (p.ndim - 1))
        elem = (1.0 - shaped) * np.exp(-nil)
        elem = np.where(inside, elem, 0.0)
        elem = np.where(np.isfinite(shaped), elem, np.nan)
        val = float(np.nanmean(elem))
        return val if np.isfinite(val) else 0.0

    return {
        "sps_raw": float(sps_raw),
        "coverage": float(coverage),
        "mean_nil": float(np.mean(nil)),
        "mean_exp_neg_nil": float(np.mean(np.exp(-nil))),
        "sps_dm": branch(dm_n),
        "sps_tke": branch(tke_n),
        "sps_mvpe": branch(mvpe_n),
        "weight_dm": SPS_WEIGHT_DM,
        "weight_tke": SPS_WEIGHT_TKE,
        "weight_mvpe": SPS_WEIGHT_MVPE,
        "normalize_factor": SPS_NORMALIZE_FACTOR,
        "sigma_global": float(official.SIGMA_GLOBAL),
    }


def evaluate_submission(submission_dir: Path, data_dir: Path, device: str = "cpu") -> dict:
    """Run the full streaming harness and return every logged metric.

    Importable entry point so notebooks (e.g. sps_bound_kaggle) reuse this
    instead of copying the loop/SPS logic. Returned dict keys: n_steps,
    n_traj, channels, mean_t, rel_l2, tke, mvpe, sps_raw, coverage,
    mean_nil, mean_exp_neg_nil, sps_dm, sps_tke, sps_mvpe, sps_score,
    subscores{rel_l2_score,tke_score,mvpe_score,time_score,sps_score},
    final_score, n_cal, mean_losses, n_losses, mean_adapt_loss.
    """
    stats_path = data_dir / "mean_std_real.pt"
    if not stats_path.exists():
        raise SystemExit(f"Missing {stats_path}. Run example_data/make_example.py first.")

    stream = build_stream(data_dir)
    normalizer = Normalizer(stats_path).to(device)

    module = import_submission(submission_dir)
    model = module.get_ttt_model(str(submission_dir), device)
    for attr in ("reset_ttt_state", "ttt_step"):
        if not hasattr(model, attr):
            raise SystemExit(f"get_ttt_model() returned an object lacking {attr}().")

    per_step_times, preds, tgts = [], [], []
    lo_raw, hi_raw = [], []  # per-step denormalized bounds
    losses: dict[str, list[float]] = {}  # every scalar info key ending in "_loss"
    prev_pair = None  # (inp_norm, tgt_norm) of the previous step
    expected_shape = None

    for step in stream:
        inp = step["input"].unsqueeze(0).to(device)     # (1, 20, 32, 64, 3)
        tgt = step["target"].unsqueeze(0).to(device)

        if step["is_first"]:
            model.reset_ttt_state()          # untimed
            prev_pair = None

        inp_norm, tgt_norm = normalizer.preprocess(inp, tgt)
        prev_target = prev_pair[1] if prev_pair is not None else None

        t0 = perf_counter()
        pred_norm, info = model.ttt_step(inp_norm, prev_target)
        per_step_times.append(perf_counter() - t0)

        prev_pair = (inp_norm.detach(), tgt_norm.detach())  # cache AFTER timing

        pred_norm = torch.as_tensor(pred_norm)
        if expected_shape is None:
            expected_shape = tuple(tgt_norm.shape)
        if tuple(pred_norm.shape) != expected_shape:
            raise SystemExit(
                f"ttt_step returned shape {tuple(pred_norm.shape)}, expected {expected_shape}."
            )
        if not isinstance(info, dict) or "adapt_loss" not in info:
            raise SystemExit('ttt_step must return (pred, info) with info["adapt_loss"].')
        # Collect all scalar losses exposed by the submission, not only the
        # contract-required adapt_loss (for example mse_loss and td_loss).
        for key, value in info.items():
            if key.endswith("_loss") and value is not None:
                try:
                    losses.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    pass

        pred_raw = normalizer.postprocess_pred(pred_norm.detach())
        preds.append(pred_raw.squeeze(0).cpu().numpy().astype(np.float32))
        tgts.append(tgt.squeeze(0).cpu().numpy().astype(np.float32))
        # Optional calibration bounds (normalized space, model device):
        # denormalize like the prediction; shape mismatch -> treat as absent.
        for key, buf in (("lower", lo_raw), ("upper", hi_raw)):
            bnd = info.get(key) if isinstance(info, dict) else None
            if bnd is not None:
                bnd = torch.as_tensor(bnd)
            if bnd is not None and tuple(bnd.shape) == expected_shape:
                buf.append(normalizer.postprocess_pred(bnd.detach().to(device))
                           .squeeze(0).cpu().numpy().astype(np.float32))
            else:
                buf.append(None)

    pred_all = np.stack(preds, axis=0)
    tgt_all = np.stack(tgts, axis=0)
    c = measured_channels(tgt_all)
    mean_t = float(np.mean(per_step_times))

    # Real subscores from the bundled scoring.py (the exact leaderboard formulas).
    official = _official_scoring()
    rl = float(np.mean(official.rel_l2_per_sample(pred_all, tgt_all, c)))
    tk = float(np.mean(official.tke_rel_l2_per_sample(pred_all, tgt_all, c)))
    mv = official.mvpe_rel_l2(pred_all, tgt_all)
    lower, upper, n_cal = resolve_sps_bands(pred_all, lo_raw, hi_raw)
    if n_cal:
        print(f"[local_eval] scoring SPS with submission intervals "
              f"({n_cal}/{len(stream)} steps; rest default band)")
    parts = sps_component_breakdown(pred_all, tgt_all, c, lower=lower, upper=upper)
    subscores = {
        "rel_l2_score": official.score_error(rl),
        "tke_score": official.score_error(tk),
        "mvpe_score": official.score_error(mv),
        "time_score": official.score_time(mean_t),
        "sps_score": official.score_sps(parts["sps_raw"]),
    }
    final = float(np.mean(list(subscores.values())))
    mean_losses = {key: float(np.mean(values)) for key, values in losses.items() if values}
    n_losses = {key: len(values) for key, values in losses.items() if values}

    return {
        "n_steps": len(stream),
        "n_traj": len({s["sim_id"] for s in stream}),
        "channels": c,
        "pred_shape": tuple(pred_all.shape),
        "mean_t": mean_t,
        "total_t": float(sum(per_step_times)),
        "rel_l2": rl,
        "tke": tk,
        "mvpe": float(mv),
        "sps_raw": parts["sps_raw"],
        "coverage": parts["coverage"],
        "mean_nil": parts["mean_nil"],
        "mean_exp_neg_nil": parts["mean_exp_neg_nil"],
        "sps_dm": parts["sps_dm"],
        "sps_tke": parts["sps_tke"],
        "sps_mvpe": parts["sps_mvpe"],
        "sps_score": subscores["sps_score"],
        "subscores": subscores,
        "final_score": final,
        "n_cal": n_cal,
        "mean_losses": mean_losses,
        "n_losses": n_losses,
        # Backward-compatible aliases used by existing notebooks.
        "mean_adapt_loss": mean_losses.get("adapt_loss"),
        "n_adapt_loss": n_losses.get("adapt_loss", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default=str(HERE),
                    help="directory containing submission.py (default: this kit)")
    ap.add_argument("--data", default=str(HERE / "example_data"),
                    help="example_data directory (default: ./example_data)")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the submission model (default: cpu; use cuda on Kaggle GPU)")
    ap.add_argument("--json-out", default=None,
                    help="optional path to write the full metrics dict as JSON")
    args = ap.parse_args()

    submission_dir = Path(args.submission).resolve()
    data_dir = Path(args.data).resolve()

    # On the tiny synthetic example_data these numbers are illustrative, NOT
    # leaderboard-comparable; point --data at real downloaded data for the true
    # metric, and note time_score reflects LOCAL wall time (hardware-dependent).
    try:
        m = evaluate_submission(submission_dir, data_dir, args.device)
    except Exception as exc:  # noqa: BLE001 - scoring is optional for the smoke test
        print(f"[local_eval] bundled scoring.py not run ({type(exc).__name__}: {exc}); "
              "shape/plumbing check still passed.")
        print("[local_eval] OK: submission ran end-to-end with correct shapes.")
        return

    print(f"[local_eval] {m['n_steps']} steps over {m['n_traj']} trajectories, batch size 1")
    print(f"[local_eval] prediction stack shape {m['pred_shape']}")
    print(f"[local_eval] mean per-step time: {m['mean_t'] * 1e3:.2f} ms "
          f"(total {m['total_t']:.3f} s over {m['n_steps']} steps)")
    print("[local_eval] real subscores from bundled scoring.py "
          "(example data, NOT leaderboard):")
    for name, val in m["subscores"].items():
        print(f"[local_eval]     {name:12s} {val:7.3f}")
    print(f"[local_eval]     {'final_score':12s} {m['final_score']:7.3f}")
    print(f"[local_eval] raw errors: rel_l2 {m['rel_l2']:.6f} "
          f"tke {m['tke']:.6f} mvpe {m['mvpe']:.6f}")
    print(f"[local_eval] sps components: raw {m['sps_raw']:.6f} "
          f"coverage {m['coverage']:.4f} mean_nil {m['mean_nil']:.4f} "
          f"mean_exp_neg_nil {m['mean_exp_neg_nil']:.4f}")
    print(f"[local_eval] sps branches: dm {m['sps_dm']:.6f} "
          f"tke {m['sps_tke']:.6f} mvpe {m['sps_mvpe']:.6f} "
          f"(weights {SPS_WEIGHT_DM:.1f}/{SPS_WEIGHT_TKE:.1f}/{SPS_WEIGHT_MVPE:.1f})")
    for name, value in m["mean_losses"].items():
        print(f"[local_eval] mean {name}: {value:.6f} "
              f"(over {m['n_losses'][name]} steps)")
    if args.json_out:
        import json
        out = Path(args.json_out)
        out.write_text(json.dumps({k: v for k, v in m.items()
                                   if k != "subscores"} | {"subscores": m["subscores"]},
                                  indent=2), encoding="utf-8")
        print(f"[local_eval] metrics written to {out}")
    print("[local_eval] OK: submission ran end-to-end with correct shapes.")


if __name__ == "__main__":
    main()
