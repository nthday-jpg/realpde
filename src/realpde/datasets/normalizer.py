"""Per-channel Gaussian normalizer for PDE velocity-field training.

The shipped baselines (CNO/FNO/Transolver) were trained in *normalized* space
— the official eval loop feeds ``ttt_step`` tensors already normalized with the
official ``mean_std_*.pt`` stats (see ``load_baseline`` docstring) — so any
model trained here must normalize too. Training on raw values silently changes
the loss scale and produces checkpoints that are inconsistent with the TTT
pipeline in ``local_eval.py``.

Format compatibility
--------------------
``save()`` writes the exact 4-tuple format of the official stats files::

    (mean_inputs, mean_targets, std_inputs, std_targets)  # each (C,)

so a file written here can be dropped straight into ``local_eval.py``'s
``Normalizer`` (and vice versa). Like that class, ``std == 0`` (e.g. the
zero-filled ``p`` channel of real data) is guarded to 1 at *use* time while
the raw 0 is preserved on disk.

No-leakage rule
---------------
Split trajectories first, fit statistics on training files/windows only, and
reuse that frozen transform for validation and test data. This mirrors the
competition evaluator, which normalizes the stream with official ``train_real``
statistics rather than statistics from hidden evaluation trajectories. The
indexed helper supports fitting after the split without materializing a subset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch


class PDENormalizer:
    """Affine per-channel normalizer: ``(x - mean) / std`` with ``std==0 -> 1``.

    Tensors have shape ``(C,)`` and broadcast over ``(..., C)`` inputs of shape
    ``(B, T, H, W, C)``. The module is intentionally *not* an ``nn.Module`` —
    it is a plain picklable container (``DataLoader`` workers, ``accelerate``)
    with explicit device handling per call.
    """

    def __init__(
        self,
        mean_in: torch.Tensor,
        mean_tgt: torch.Tensor,
        std_in: torch.Tensor,
        std_tgt: torch.Tensor,
    ):
        self.mean_in = mean_in.detach().float().reshape(-1)
        self.mean_tgt = mean_tgt.detach().float().reshape(-1)
        self.std_in = std_in.detach().float().reshape(-1)
        self.std_tgt = std_tgt.detach().float().reshape(-1)

    # -- construction ------------------------------------------------------
    @classmethod
    def fit_from_samples(
        cls,
        inputs: Sequence[torch.Tensor] | torch.Tensor,
        targets: Sequence[torch.Tensor] | torch.Tensor,
    ) -> "PDENormalizer":
        """Fit per-channel mean/std over all sample/space/time dims.

        Accepts either stacked ``(N, T, H, W, C)`` tensors or sequences of
        ``(T, H, W, C)`` sample tensors (e.g. ``[ds[i][0] for i in idx]``).
        Accumulates in float64 for stability; never materializes the full
        stack when given a sequence.
        """
        n_in = s_in = ss_in = None
        n_t = s_t = ss_t = None
        c_in = c_t = None

        def _acc(x: torch.Tensor, kind: str):
            nonlocal n_in, s_in, ss_in, n_t, s_t, ss_t, c_in, c_t
            xf = x.double().reshape(-1, x.shape[-1])  # (M, C)
            if kind == "in":
                c_in = xf.shape[-1]
                s = xf.sum(0)
                ss = (xf * xf).sum(0)
                n = xf.shape[0]
                s_in = s if s_in is None else s_in + s
                ss_in = ss if ss_in is None else ss_in + ss
                n_in = n if n_in is None else n_in + n
            else:
                c_t = xf.shape[-1]
                s = xf.sum(0)
                ss = (xf * xf).sum(0)
                n = xf.shape[0]
                s_t = s if s_t is None else s_t + s
                ss_t = ss if ss_t is None else ss_t + ss
                n_t = n if n_t is None else n_t + n

        if torch.is_tensor(inputs):
            _acc(inputs, "in")
        else:
            for x in inputs:
                _acc(torch.as_tensor(x), "in")
        if torch.is_tensor(targets):
            _acc(targets, "tgt")
        else:
            for y in targets:
                _acc(torch.as_tensor(y), "tgt")

        mean_in = (s_in / n_in).float()
        mean_tgt = (s_t / n_t).float()
        var_in = (ss_in / n_in - (s_in / n_in) ** 2).clamp_min(0)
        var_tgt = (ss_t / n_t - (s_t / n_t) ** 2).clamp_min(0)
        return cls(mean_in, mean_tgt, var_in.sqrt().float(), var_tgt.sqrt().float())

    @classmethod
    def fit_from_indexed(cls, dataset, indices: Iterable[int]) -> "PDENormalizer":
        """Fit on ``dataset[i] -> (input, target)`` pairs for ``indices`` only.

        Streams one sample at a time (no stacking), so it is safe for the full
        ``train_sim`` split. Pass training indices after splitting, then reuse
        the returned normalizer for validation and test data.
        """
        idx = list(indices)
        return cls.fit_from_samples(
            (torch.as_tensor(dataset[i][0]) for i in idx),
            (torch.as_tensor(dataset[i][1]) for i in idx),
        )

    # -- persistence (official mean_std_*.pt layout) ------------------------
    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.mean_in, self.mean_tgt, self.std_in, self.std_tgt)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.parent != Path("") and str(path.parent) not in (".", ""):
            path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.as_tuple(), path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PDENormalizer":
        mi, mt, si, st = torch.load(path, map_location="cpu", weights_only=False)
        return cls(mi, mt, si, st)

    @classmethod
    def from_ckpt(cls, ckpt_obj: dict, key: str = "norm_val") -> "PDENormalizer | None":
        """Pull a saved normalizer out of a trainer checkpoint dict.

        Returns ``None`` when the checkpoint predates normalization.
        """
        tup = (ckpt_obj or {}).get(key)
        if tup is None:
            return None
        mi, mt, si, st = tup
        return cls(
            torch.as_tensor(mi), torch.as_tensor(mt),
            torch.as_tensor(si), torch.as_tensor(st),
        )

    # -- use -----------------------------------------------------------------
    @staticmethod
    def _safe_std(std: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        std = std.to(device=like.device, dtype=like.dtype)
        return torch.where(std == 0, torch.ones_like(std), std)

    def preprocess(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize an ``(input, target)`` batch pair (any leading dims)."""
        c1, c2 = x.shape[-1], y.shape[-1]
        mi = self.mean_in.to(device=x.device, dtype=x.dtype)[..., :c1]
        si = self._safe_std(self.std_in[..., :c1], x)
        mt = self.mean_tgt.to(device=y.device, dtype=y.dtype)[..., :c2]
        st = self._safe_std(self.std_tgt[..., :c2], y)
        return (x - mi) / si, (y - mt) / st

    def postprocess_pred(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """Map a normalized prediction back to raw space (for metrics)."""
        c = pred_norm.shape[-1]
        mt = self.mean_tgt.to(device=pred_norm.device, dtype=pred_norm.dtype)[..., :c]
        st = self._safe_std(self.std_tgt[..., :c], pred_norm)
        return pred_norm * st + mt

    def describe(self) -> str:
        fmt = lambda t: "[" + ", ".join(f"{v:.4f}" for v in t.reshape(-1).tolist()) + "]"
        return (f"mean_in={fmt(self.mean_in)} std_in={fmt(self.std_in)} "
                f"mean_tgt={fmt(self.mean_tgt)} std_tgt={fmt(self.std_tgt)}")


__all__ = ["PDENormalizer"]
