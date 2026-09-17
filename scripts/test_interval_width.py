#!/usr/bin/env python3
"""Answer: is the default SPS interval band (0.05 * |pred|) too narrow?

Runs the frozen CNO model over data with a fixed RELATIVE interval band of a
given width, sweeps widths, and reports the official SPS score + coverage for
each. This isolates the sharpness-vs-coverage tradeoff of the width knob.

USAGE
    python scripts/test_interval_width.py example   # tiny synthetic (has stats)
    python scripts/test_interval_width.py real       # full train_real (stats computed)
    python scripts/test_interval_width.py real --ckpt data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth

For "real", the local affine stats are computed from train_real (u,v scaled by
their pooled std; p left at zero-scale). This is a local approximation of the
competition's mean_std_real.pt; numbers are directional, not leaderboard.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scoring as official  # noqa: E402

IN_STEP = 20
OUT_STEP = 20
INTERVAL = 20
SUB_S = 2
HORIZON = IN_STEP + OUT_STEP


def build_stream(data_dir: Path):
    """Data dir has test_real/*.h5 + mean_std_real.pt (example layout)."""
    test_dir = data_dir / "test_real"
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".h5"))
    return _stream_from_files(files_dir=test_dir, frame_names=files)


def build_stream_flat(data_dir: Path):
    """Data dir is a flat directory of .h5 files (train_real layout)."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".h5"))
    return _stream_from_files(files_dir=data_dir, frame_names=files)


def _stream_from_files(files_dir: Path, frame_names: list[str]):
    entries, frames_of = [], {}
    for name in frame_names:
        with h5py.File(files_dir / name, "r") as f:
            frames_of[name] = f["u"].shape[0]
        for t in range(0, frames_of[name] - HORIZON + 1, INTERVAL):
            entries.append((name, t))
    entries.sort(key=lambda e: (e[0], e[1]))
    stream, prev = [], None
    for name, tid in entries:
        with h5py.File(files_dir / name, "r") as f:
            end = min(tid + HORIZON, frames_of[name])
            u = f["u"][tid:end, ::SUB_S, ::SUB_S]
            v = f["v"][tid:end, ::SUB_S, ::SUB_S]
        p = np.zeros_like(u)
        data = np.stack([u, v, p], axis=-1)
        if data.shape[0] < HORIZON:
            data = np.concatenate([data, np.repeat(data[-1:], HORIZON - data.shape[0], 0)], 0)
        stream.append({
            "input": data[:IN_STEP], "target": data[IN_STEP:HORIZON],
            "is_first": name != prev,
        })
        prev = name
    return stream


class Normalizer:
    def __init__(self, sp: Path):
        mi, mt, si, st = torch.load(sp, map_location="cpu", weights_only=False)
        one = torch.ones_like
        self.mean_in, self.mean_tgt = mi.float(), mt.float()
        self.std_in = torch.where(si == 0, one(si), si).float()
        self.std_tgt = torch.where(st == 0, one(st), st).float()

    def preprocess(self, x, y):
        c1, c2 = x.shape[-1], y.shape[-1]
        return ((x - self.mean_in[..., :c1]) / self.std_in[..., :c1],
                (y - self.mean_tgt[..., :c2]) / self.std_tgt[..., :c2])

    def post(self, p):
        c = p.shape[-1]
        return p * self.std_tgt[..., :c] + self.mean_tgt[..., :c]


class NFlt:
    def __init__(self, mt, st):
        self.mean_tgt = torch.tensor(mt, dtype=torch.float32)
        self.std_tgt = torch.tensor(st, dtype=torch.float32)

    def preprocess(self, x, y):
        return (x - self.mean_tgt) / self.std_tgt, (y - self.mean_tgt) / self.std_tgt

    def post(self, p):
        return p * self.std_tgt + self.mean_tgt


def build_flat_normalizer(flat_dir: Path, n_files: int = 2):
    files = sorted(f for f in os.listdir(flat_dir) if f.endswith(".h5"))
    vals = []
    for name in files[:n_files]:
        with h5py.File(flat_dir / name, "r") as f:
            u = f["u"][...].astype(np.float32)
            v = f["v"][...].astype(np.float32)
        # subsample frames to bound memory
        u = u[:: (max(1, len(u) // 20))].reshape(-1)
        v = v[:: (max(1, len(v) // 20))].reshape(-1)
        vals.append(u); vals.append(v)
    allv = np.concatenate(vals)
    mu, sd = float(np.mean(allv)), float(np.std(allv))
    mt = np.array([mu, mu, 0.0], dtype=np.float32).reshape(1, 1, 1, 1, 3)
    st = np.array([sd, sd, 1.0], dtype=np.float32).reshape(1, 1, 1, 1, 3)
    return NFlt(mt, st)


def measured_channels(t: np.ndarray) -> int:
    return max(1, sum(not np.allclose(t[..., i], 0.0) for i in range(t.shape[-1])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["example", "real"], default="example", nargs="?")
    ap.add_argument("--ckpt", default=None,
                    help="real CNO checkpoint to stage as model.pth (else fallback)")
    ap.add_argument("--history", type=int, default=5)
    args = ap.parse_args()

    if args.mode == "real":
        data_path = REPO / "data" / "train_real"
        stream = build_stream_flat(data_path)
        norm = build_flat_normalizer(data_path)
    else:
        data_path = REPO / "example_data"
        stream = build_stream(data_path)
        norm = Normalizer(data_path / "mean_std_real.pt")

    sub_dir = REPO / "submissions" / "submission_v5"
    if args.ckpt:
        staged = sub_dir / "model.pth"
        copy = False
        if not staged.exists():
            import shutil
            shutil.copy(args.ckpt, staged)
            copy = True
    sys.path.insert(0, str(sub_dir))
    import submission as v5  # noqa
    mod = v5.get_ttt_model(str(sub_dir), "cpu")
    if args.ckpt and copy:
        staged.unlink()

    preds, tgts = [], []
    prev_pair = None
    for s in stream:
        inp = torch.tensor(s["input"], dtype=torch.float32).unsqueeze(0)
        tgt = torch.tensor(s["target"], dtype=torch.float32).unsqueeze(0)
        if s["is_first"]:
            mod.reset_ttt_state(); prev_pair = None
        in_n, tg_n = norm.preprocess(inp, tgt)
        prev_t = prev_pair[1] if prev_pair else None
        pred_n, _ = mod.ttt_step(in_n, prev_t)
        prev_pair = (in_n, tg_n)
        preds.append(norm.post(torch.as_tensor(pred_n).detach()).squeeze(0).numpy().astype(np.float32))
        tgts.append(tgt.squeeze(0).numpy().astype(np.float32))
    pred_all = np.stack(preds, 0); tgt_all = np.stack(tgts, 0)
    c = measured_channels(tgt_all)
    print(f"[test] {len(stream)} steps, measured={c}, pred {pred_all.shape}, mode={args.mode}")

    print(f"\n{'band(half-frac)':>16}{'total-width':>13}{'sps_score':>12}{'coverage':>10}{'mean_nil':>10}")
    for half in [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]:
        lo = pred_all - half * np.abs(pred_all)
        hi = pred_all + half * np.abs(pred_all)
        sps, cov = official.aggregate_sps(pred_all, tgt_all, c, lower=lo, upper=hi)
        nil = ((hi - lo) / official.SIGMA_GLOBAL)[..., :c]
        print(f"{half:>16.3f}{2*half:>13.3f}{official.score_sps(sps):>12.2f}"
              f"{cov:>10.3f}{float(np.nanmean(nil)):>10.2f}")


if __name__ == "__main__":
    main()