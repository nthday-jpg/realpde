from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_one_file(
    path: Path, in_step: int, out_step: int, interval: int, sub_s: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load a single .h5 file into (inputs, targets) window tensors.

    Thread-safe: opens its own h5py handle, touches no shared state.
    """
    with h5py.File(path, "r") as f:
        u = np.asarray(f["u"][:, ::sub_s, ::sub_s], dtype=np.float32)
        v = np.asarray(f["v"][:, ::sub_s, ::sub_s], dtype=np.float32)

    p = np.zeros_like(u)
    data = np.stack([u, v, p], axis=-1)  # (T, H, W, 3)
    T = data.shape[0]

    inputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for t in range(0, T - in_step - out_step + 1, interval):
        inputs.append(torch.from_numpy(data[t : t + in_step].copy()))
        targets.append(torch.from_numpy(data[t + in_step : t + in_step + out_step].copy()))
    return inputs, targets


class PDEDataset(Dataset):
    """HDF5 dataset for PDE velocity field sequences with full preloading.

    Loads all ``.h5`` files into memory at init time for fast training.
    Each file contains ``u``, ``v``, ``p`` datasets of shape
    ``(T, H_native, W_native)``.  Spatial subsampling (``sub_s``) is applied
    at load time to reduce native ``64 x 128`` PIV grids to ``32 x 64``.

    Each sample is a ``(input, target)`` pair where:

    - ``input``  has shape ``(in_step,  32, 64, 3)``
    - ``target`` has shape ``(out_step, 32, 64, 3)``

    Parameters
    ----------
    data_dir : str | Path
        Directory containing ``.h5`` files.
    in_step : int
        Number of input frames.
    out_step : int
        Number of target frames.
    interval : int
        Temporal stride between consecutive samples from the same file.
    sub_s : int
        Spatial subsampling factor (2 = downsample 64x128 to 32x64).
    preload_workers : int | None
        Threads for parallel per-file preload (default: ``os.cpu_count()``
        or ``PDE_PRELOAD_WORKERS`` env). 0/1 = serial. Capped at n_files;
        ``executor.map`` preserves file order so sample order is deterministic.
    """

    def __init__(
        self,
        data_dir: str | Path,
        in_step: int = 20,
        out_step: int = 20,
        interval: int = 20,
        sub_s: int = 2,
        preload_workers: int | None = None,
    ):
        self.in_step = in_step
        self.out_step = out_step
        self.interval = interval

        data_dir = Path(data_dir)
        h5_files = sorted(f for f in data_dir.iterdir() if f.suffix == ".h5")
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {data_dir}")

        if preload_workers is None:
            env = os.environ.get("PDE_PRELOAD_WORKERS", "").strip()
            if env != "":
                try:
                    preload_workers = int(env)
                except ValueError:
                    preload_workers = 0
            else:
                preload_workers = os.cpu_count() or 4
        # 0/negative = serial fallback; cap at n_files to avoid idle threads
        workers = min(max(int(preload_workers), 0), len(h5_files))

        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

        if workers <= 1:
            results = [
                _load_one_file(p, in_step, out_step, interval, sub_s)
                for p in h5_files
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                # executor.map preserves input order -> deterministic sample order
                results = list(
                    ex.map(
                        _load_one_file,
                        h5_files,
                        [in_step] * len(h5_files),
                        [out_step] * len(h5_files),
                        [interval] * len(h5_files),
                        [sub_s] * len(h5_files),
                    )
                )

        for ins, tgts in results:
            self.inputs.extend(ins)
            self.targets.extend(tgts)

        if not self.inputs:
            raise FileNotFoundError(
                f"No valid samples found in {data_dir} "
                f"(need at least {in_step + out_step} frames per file)"
            )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
