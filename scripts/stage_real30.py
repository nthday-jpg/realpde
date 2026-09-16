#!/usr/bin/env python3
"""Stage a real-data subset for local_eval: sample trajectories, truncate, stats.

- Samples N files (seeded), keeps the first F frames of each.
- Writes float32 truncated copies to <dst>/test_real/*.h5 (full float64
  files are ~60-90MB each; truncation keeps staging fast).
- Computes protocol-exact (input-window / target-window, ::2 subsampled)
  per-channel mean/std and saves the (mean_in, mean_tgt, std_in, std_tgt)
  tuple as <dst>/mean_std_real.pt.

Usage (local, CPU smoke of the plumbing on 2 files):
    python scripts/stage_real30.py --src data/train_real --dst /tmp/real30 \\
        --n-files 2 --frames 60

Usage (Kaggle, GPU run of 30 trajectories; resolve --src from
DATA_ROOT=/kaggle/input/datasets/nthday/realpde as in notebook/eval_kaggle.ipynb):
    python scripts/stage_real30.py --src <train_real dir holding *.h5> \\
        --dst /kaggle/working/real30 --n-files 30 --frames 200
    python local_eval.py --submission submissions/submission_v3 \\
        --data /kaggle/working/real30 --device cuda
"""
import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch

IN_STEP, OUT_STEP, INTERVAL, SUB_S = 20, 20, 20, 2
HORIZON = IN_STEP + OUT_STEP


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data/train_real",
                    help="dir of source .h5 trajectories")
    ap.add_argument("--dst", default="/tmp/real30",
                    help="staging root; writes <dst>/test_real/*.h5 + <dst>/mean_std_real.pt")
    ap.add_argument("--n-files", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=200,
                    help="keep first N frames per trajectory "
                         "(windows/file = (N-40)/20+1 under the streaming protocol)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) / "test_real"
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p.name for p in src.glob("*.h5"))
    if len(files) < args.n_files:
        raise SystemExit(f"[error] only {len(files)} .h5 under {src}, want {args.n_files}")
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(files, args.n_files))
    print(f"[stage] sampled {len(chosen)} files from {src} (seed={args.seed}):")
    for name in chosen:
        print(f"  {name}")

    # Accumulators for window-based stats (subsampled, u/v only; p is zero-filled).
    sum_in = np.zeros(2, dtype=np.float64)
    sumsq_in = np.zeros(2, dtype=np.float64)
    sum_tg = np.zeros(2, dtype=np.float64)
    sumsq_tg = np.zeros(2, dtype=np.float64)
    n_in = n_tg = 0

    for name in chosen:
        with h5py.File(src / name, "r") as f:
            u = f["u"][:args.frames].astype(np.float32)
            v = f["v"][:args.frames].astype(np.float32)
        with h5py.File(dst / name, "w") as f:
            f.create_dataset("u", data=u, compression="lzf")
            f.create_dataset("v", data=v, compression="lzf")
        us = u[:, ::SUB_S, ::SUB_S].astype(np.float64)
        vs = v[:, ::SUB_S, ::SUB_S].astype(np.float64)
        for t in range(0, args.frames - HORIZON + 1, INTERVAL):
            inp = np.stack([us[t:t + IN_STEP], vs[t:t + IN_STEP]], axis=-1)
            tgt = np.stack([us[t + IN_STEP:t + HORIZON], vs[t + IN_STEP:t + HORIZON]], axis=-1)
            sum_in += inp.sum(axis=(0, 1, 2)); sumsq_in += (inp ** 2).sum(axis=(0, 1, 2))
            sum_tg += tgt.sum(axis=(0, 1, 2)); sumsq_tg += (tgt ** 2).sum(axis=(0, 1, 2))
            n_in += inp.shape[0] * inp.shape[1] * inp.shape[2]
            n_tg += tgt.shape[0] * tgt.shape[1] * tgt.shape[2]

    mean_in = sum_in / n_in
    mean_tg = sum_tg / n_tg
    std_in = np.sqrt(np.maximum(sumsq_in / n_in - mean_in ** 2, 1e-12))
    std_tg = np.sqrt(np.maximum(sumsq_tg / n_tg - mean_tg ** 2, 1e-12))
    print(f"[stage] mean_in {mean_in} std_in {std_in}")
    print(f"[stage] mean_tg {mean_tg} std_tg {std_tg}")

    mi = torch.tensor([*mean_in, 0.0], dtype=torch.float32)
    mt = torch.tensor([*mean_tg, 0.0], dtype=torch.float32)
    si = torch.tensor([*std_in, 1.0], dtype=torch.float32)
    st = torch.tensor([*std_tg, 1.0], dtype=torch.float32)
    torch.save((mi, mt, si, st), dst.parent / "mean_std_real.pt")
    per_file = len(range(0, args.frames - HORIZON + 1, INTERVAL))
    print(f"[stage] wrote {dst.parent / 'mean_std_real.pt'}; "
          f"{per_file} windows/file x {len(chosen)} files = {per_file * len(chosen)} steps")


if __name__ == "__main__":
    main()
