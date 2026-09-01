from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PDEDataset(Dataset):
    """HDF5 dataset for PDE velocity field sequences.

    Loads ``.h5`` files containing ``u``, ``v``, ``p`` datasets each of shape
    ``(T, H_native, W_native)``.  Spatial subsampling (``sub_s``) is applied at
    load time to reduce native ``64 x 128`` PIV grids to ``32 x 64``.

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
    """

    def __init__(
        self,
        data_dir: str | Path,
        in_step: int = 20,
        out_step: int = 20,
        interval: int = 20,
        sub_s: int = 2,
    ):
        self.in_step = in_step
        self.out_step = out_step
        self.interval = interval
        self.sub_s = sub_s

        data_dir = Path(data_dir)
        h5_files = sorted(f for f in data_dir.iterdir() if f.suffix == ".h5")
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {data_dir}")

        self.samples: List[Tuple[Path, int]] = []
        self.file_lengths: dict[Path, int] = {}
        for path in h5_files:
            with h5py.File(path, "r") as f:
                n_frames = f["u"].shape[0]
            self.file_lengths[path] = n_frames
            max_time_id = n_frames - (in_step + out_step)
            for t in range(0, max_time_id + 1, interval):
                self.samples.append((path, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, t = self.samples[idx]
        with h5py.File(path, "r") as f:
            u = f["u"][t : t + self.in_step + self.out_step, :: self.sub_s, :: self.sub_s]
            v = f["v"][t : t + self.in_step + self.out_step, :: self.sub_s, :: self.sub_s]

        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        p = np.zeros_like(u)
        data = np.stack([u, v, p], axis=-1)  # (T, H, W, 3)

        inp = torch.from_numpy(data[: self.in_step])
        tgt = torch.from_numpy(data[self.in_step : self.in_step + self.out_step])
        return inp, tgt
