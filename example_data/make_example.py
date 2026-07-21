#!/usr/bin/env python3
"""Generate the tiny synthetic example_data for local smoke tests (reproducible).

Writes two short trajectories under ``test_real/`` and copies the official
normalization stats next to them, matching the layout expected by
``CompetitionAirfoil(mode="test", dataset_type="real")``:

    example_data/
      test_real/
        <name>.h5          # flat u, v, p datasets, native 64x128, float32
      mean_std_real.pt     # copied from bundle_track2/data/

Each file has 60 frames, giving 2 streaming steps per trajectory (time_ids
0 and 20 at interval 20, horizon 40) so the prev-target adaptation path is
exercised. The fields are smooth advecting sinusoids at the real u/v scale --
purely synthetic, NOT physical PIV data. Total size stays well under 5 MB.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEST_DIR = HERE / "test_real"
STATS_DST = HERE / "mean_std_real.pt"
STATS_SRC = HERE / ".." / ".." / "data" / "mean_std_real.pt"

# Native resolution (subsampled by 2 at load time -> 32x64) and frame count.
H, W, N_FRAMES = 64, 128, 60

# Two trajectories: names hint at the real {Re}_{AoA}.h5 convention.
CONFIGS = [
    # (filename, kx, ky, omega, phase)
    ("2968_0.h5", 2.0, 1.0, 0.20, 0.0),
    ("5025_5.h5", 3.0, 2.0, 0.28, 1.3),
]


def load_scale():
    """Per-channel (mean, std) for u, v from the official stats (p stays 0)."""
    if STATS_SRC.exists():
        shutil.copy(STATS_SRC, STATS_DST)
    mean_in, _, std_in, _ = torch.load(STATS_DST, weights_only=False)
    mean = mean_in.reshape(-1).float().numpy()
    std = std_in.reshape(-1).float().numpy()
    return mean, std


def make_field(kx, ky, omega, phase, mean, std):
    """Smooth advecting sinusoids at the u/v scale; shape (N_FRAMES, H, W, 3).

    Deterministic (no additive noise) so the fields stay smooth and compress
    well; the traveling waves still give non-degenerate TKE / MVPE. Stored as
    float16 (the loader upcasts to float32) to keep the bundle small.
    """
    yy = np.linspace(0.0, 1.0, H)[None, :, None]
    xx = np.linspace(0.0, 1.0, W)[None, None, :]
    t = np.arange(N_FRAMES)[:, None, None]
    u = mean[0] + std[0] * np.sin(2 * np.pi * (kx * xx + ky * yy) - omega * t + phase)
    v = mean[1] + std[1] * np.cos(2 * np.pi * (ky * xx + kx * yy) - omega * t + phase)
    # Smooth temporal/spatial modulation so TKE / MVPE are non-degenerate.
    u = u + 0.3 * std[0] * np.cos(2 * np.pi * yy + 0.1 * t)
    v = v + 0.3 * std[1] * np.sin(2 * np.pi * xx + 0.1 * t)
    p = np.zeros_like(u)
    field = np.stack([np.broadcast_to(u, (N_FRAMES, H, W)),
                      np.broadcast_to(v, (N_FRAMES, H, W)),
                      p], axis=-1)
    return field.astype(np.float16)


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    mean, std = load_scale()
    for name, kx, ky, omega, phase in CONFIGS:
        field = make_field(kx, ky, omega, phase, mean, std)  # (T, H, W, 3)
        out = TEST_DIR / name
        with h5py.File(out, "w") as f:
            for i, ch in enumerate(("u", "v", "p")):
                f.create_dataset(ch, data=field[..., i], compression="gzip", compression_opts=9)
        print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB, {N_FRAMES} frames {H}x{W})")
    if STATS_DST.exists():
        print(f"stats: {STATS_DST} ({os.path.getsize(STATS_DST)} bytes)")
    total = sum(os.path.getsize(TEST_DIR / c[0]) for c in CONFIGS) + (
        os.path.getsize(STATS_DST) if STATS_DST.exists() else 0)
    print(f"total example_data size: {total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
