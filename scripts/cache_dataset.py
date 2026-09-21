#!/usr/bin/env python3
"""Cache a windowed PDE dataset to one .pt file for fast reuse on Kaggle.

PDEDataset reads every .h5 through h5py (slow, every run). This dumps all
(in, out) windows once to a stacked .pt in /kaggle/working (persists across
sessions of the same notebook), so later runs just torch.load it.

Run once per split (plain python, CPU is fine):
    python scripts/cache_dataset.py            # env: DATA_PATH, CACHE_OUT, window knobs

Config via env:
    DATA_PATH      dir with .h5 files
    CACHE_OUT      output .pt path, e.g. /kaggle/working/cache_train_sim.pt
    CACHE_WORKERS  HDF5 loader processes (default: all available CPUs)
    IN_STEP/OUT_STEP/INTERVAL/SUB_S (default 20/20/20/2)

~5GB fp32 for the full train_sim split (4900 windows x 2 x 20x32x64x3).
"""
from __future__ import annotations

import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realpde.datasets import PDEDataset, file_window_counts


def main():
    data_path = os.environ.get("DATA_PATH", "data/train_sim")
    cache_out = os.environ.get("CACHE_OUT", "/kaggle/working/cache_train_sim.pt")
    # h5py serializes much of threaded I/O, which leaves Kaggle CPUs mostly
    # idle. Separate processes provide actual per-file parallelism.
    os.environ["PDE_PRELOAD_BACKEND"] = "process"
    os.environ["PDE_PRELOAD_WORKERS"] = os.environ.get(
        "CACHE_WORKERS", str(os.cpu_count() or 4))
    kw = dict(in_step=int(os.environ.get("IN_STEP", 20)),
              out_step=int(os.environ.get("OUT_STEP", 20)),
              interval=int(os.environ.get("INTERVAL", 20)),
              sub_s=int(os.environ.get("SUB_S", 2)))

    if os.path.exists(cache_out):
        print(f"[cache] exists, skipping: {cache_out}")
        return

    t0 = time.time()
    print(f"[cache] loading with {os.environ['PDE_PRELOAD_WORKERS']} processes")
    ds = PDEDataset(data_path, **kw)
    print(f"[cache] {len(ds)} windows from {data_path} "
          f"({time.time()-t0:.0f}s to read) — stacking...")
    ins, tgts = [], []
    for i in range(len(ds)):
        inp, tgt = ds[i]
        ins.append(inp)
        tgts.append(tgt)
        if (i + 1) % 1000 == 0:
            print(f"[cache] stacked {i+1}/{len(ds)}")
    counts = file_window_counts(data_path, in_step=kw["in_step"],
                                out_step=kw["out_step"], interval=kw["interval"])
    assert sum(n for _, n in counts) == len(ds), "counts disagree with dataset"
    blob = {"inputs": torch.stack(ins), "targets": torch.stack(tgts),
            "data_path": str(data_path), "n": len(ds),
            "file_names": [name for name, _ in counts],
            "file_counts": [[name, n] for name, n in counts], **kw}
    del ins, tgts
    os.makedirs(os.path.dirname(cache_out) or ".", exist_ok=True)
    torch.save(blob, cache_out)
    gb = os.path.getsize(cache_out) / 1024**3
    print(f"[cache] wrote {cache_out} ({gb:.1f} GB, "
          f"inputs {tuple(blob['inputs'].shape)}) in {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
