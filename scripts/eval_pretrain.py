#!/usr/bin/env python3
"""Direct (non-TTT) eval of a pretrained checkpoint.

Scores a UNet (or any checkpoint saved by ``scripts/trainer.py``) on a
PDEDataset split — reports MSE and relative-L2 in *raw* space (the same
formula the leaderboard uses). The model runs in normalized space (per-split
stats, fit on the eval split only — no train stats leak in) and predictions
are denormalized before scoring, so ``val_loss``-style normalized numbers and
raw leaderboard-style numbers are both reported. Optionally compares against
the CNO baseline from /kaggle/input/realpde/baseline/.

The val split matches the trainer exactly: trajectory (file-level) holdout via
``trajectory_split_indices`` with the same ``VAL_FRAC``/``SEED`` — not the old
"last 10% of windows" approximation.

Usage in Kaggle (after `accelerate launch scripts/trainer.py`):
    !python scripts/eval_pretrain.py --ckpt /kaggle/working/checkpoints/best.pth
    !python scripts/eval_pretrain.py --ckpt /kaggle/working/checkpoints/best.pth --baseline /kaggle/input/realpde/baseline/sim_real_cno.pth

Locally:
    python scripts/eval_pretrain.py --ckpt checkpoints/best.pth --data data/train_sim --split val
    python scripts/eval_pretrain.py --ckpt data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth --baseline-model cno --data example_data/test_real --sub-s 2
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent  # scripts/ -> repo root
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realpde.datasets import PDEDataset, PDENormalizer, file_window_counts, trajectory_split_indices


def _infer_unet_cfg(ckpt_obj, args):
    cfg = ckpt_obj.get("cfg", {}) if isinstance(ckpt_obj, dict) else {}
    in_step = int(cfg.get("in_step", args.in_step))
    out_step = int(cfg.get("out_step", args.out_step))
    channels = int(cfg.get("unet_channels", cfg.get("channels", args.channels)))
    n_layers = int(cfg.get("unet_n_layers", cfg.get("n_layers", args.n_layers)))
    return dict(in_step=in_step, out_step=out_step, channels=channels, n_layers=n_layers)


def build_unet_from_ckpt(ckpt_path: Path, device: str):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    cfg = _infer_unet_cfg(obj, argparse.Namespace(in_step=20, out_step=20, channels=16, n_layers=2))
    from realpde.models.unet import UNet, UNetConfig
    model_cfg = UNetConfig(in_step=cfg["in_step"], out_step=cfg["out_step"], channels=cfg["channels"], n_layers=cfg["n_layers"])
    model = UNet(model_cfg)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, obj.get("cfg", {}), cfg


def rel_l2_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    # pred/target: (B, T, H, W, C) -> per-sample rel L2 over flattened (T*H*W*C) for active channels
    b = pred.shape[0]
    p = pred.reshape(b, -1)
    t = target.reshape(b, -1)
    denom = torch.linalg.norm(t, dim=1).clamp_min(1e-8)
    return (torch.linalg.norm(p - t, dim=1) / denom).mean().item()


def main():
    ap = argparse.ArgumentParser(description="Direct eval of pretrained UNet checkpoint")
    ap.add_argument("--ckpt", type=str, required=True, help="path to best.pth / final.pth (scripts/trainer.py output)")
    ap.add_argument("--data", type=str, default=None, help="override DATA_PATH (default: $DATA_PATH or data/train_sim)")
    ap.add_argument("--split", choices=["val", "full"], default="val", help="val = trajectory holdout (same as trainer), full = all samples")
    ap.add_argument("--val-frac", type=float, default=None, help="val file fraction (default: ckpt cfg VAL_FRAC or 0.1)")
    ap.add_argument("--seed", type=int, default=None, help="trajectory split seed (default: ckpt cfg SEED or 42)")
    ap.add_argument("--use-ckpt-stats", action="store_true", help="use norm_val saved in the ckpt instead of fitting stats on the eval split")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--in-step", type=int, default=20)
    ap.add_argument("--out-step", type=int, default=20)
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--sub-s", type=int, default=2)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--baseline", type=str, default=None, help="optional CNO/FNO/Transolver checkpoint for comparison (e.g. /kaggle/input/realpde/baseline/sim_real_cno.pth)")
    ap.add_argument("--baseline-model", type=str, default=None, choices=["cno","fno","transolver"], help="force baseline type (avoids filename detection for model.pth)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_path = args.data or os.environ.get("DATA_PATH") or "data/train_sim"
    data_path = Path(data_path)
    if not data_path.exists():
        # try Kaggle nested fallback
        candidates = [Path("/kaggle/input/realpde/train_sim/train_sim"), Path("/kaggle/input/realpde/train_sim")]
        for c in candidates:
            if c.exists() and any(c.glob("*.h5")):
                data_path = c
                break
    print(f"[eval] data  : {data_path}")
    print(f"[eval] ckpt  : {args.ckpt}")
    print(f"[eval] device: {args.device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    # Build dataset with interval/sub_s matching training (ckpt cfg overrides defaults)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("cfg", {}) if isinstance(obj, dict) else {}
    in_step = int(cfg.get("in_step", args.in_step))
    out_step = int(cfg.get("out_step", args.out_step))
    interval = int(cfg.get("interval", args.interval))
    sub_s = int(cfg.get("sub_s", args.sub_s))
    print(f"[eval] cfg from ckpt: in_step={in_step} out_step={out_step} interval={interval} sub_s={sub_s}")

    full_ds = PDEDataset(data_path, in_step=in_step, out_step=out_step, interval=interval, sub_s=sub_s)
    print(f"[eval] dataset: {len(full_ds)} samples from {data_path} (in {in_step} -> out {out_step}, interval {interval}, sub_s {sub_s})")

    if args.split == "val":
        # Same trajectory holdout the trainer uses (VAL_FRAC of FILES, seeded).
        val_frac = args.val_frac if args.val_frac is not None else float(cfg.get("val_frac", os.environ.get("VAL_FRAC", 0.1)))
        seed = args.seed if args.seed is not None else int(cfg.get("seed", os.environ.get("SEED", 42)))
        counts = file_window_counts(data_path, in_step, out_step, interval)
        assert sum(n for _, n in counts) == len(full_ds), "file counts disagree with dataset length"
        _, val_idx = trajectory_split_indices(counts, val_frac, seed)
        val_ds = Subset(full_ds, val_idx)
        print(f"[eval] split: val ({len(val_idx)}/{len(full_ds)} windows by trajectory, frac={val_frac} seed={seed})")
    else:
        val_idx = list(range(len(full_ds)))
        val_ds = full_ds
        print(f"[eval] split: full ({len(full_ds)} samples)")

    # --- Normalization: fit on the EVAL split only (no train leakage) --------
    # The model was trained in normalized space, so eval inputs must be
    # normalized too; predictions are denormalized before raw-space scoring.
    norm = None
    if args.use_ckpt_stats:
        norm = PDENormalizer.from_ckpt(obj, "norm_val")
        if norm is None:
            print("[eval] WARNING: --use-ckpt-stats but ckpt has no norm_val; fitting on eval split instead")
    if norm is None:
        norm = PDENormalizer.fit_from_indexed(full_ds, val_idx)
        print(f"[eval] norm (fit on eval {args.split} split): {norm.describe()}")
    else:
        print(f"[eval] norm (from ckpt norm_val): {norm.describe()}")

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model, _, _ = build_unet_from_ckpt(ckpt_path, args.device)
    print(f"[eval] loaded UNet: {model.__class__.__name__} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")

    # Optional baseline
    baseline = None
    if args.baseline:
        bpath = Path(args.baseline)
        if not bpath.exists():
            print(f"[eval] WARNING: baseline not found: {bpath} — skipping comparison")
        else:
            btype = args.baseline_model
            if btype is None:
                # infer from filename; model.pth needs explicit type
                name = bpath.name.lower()
                if "cno" in name: btype = "cno"
                elif "fno" in name: btype = "fno"
                elif "transolver" in name: btype = "transolver"
                else:
                    print(f"[eval] WARNING: cannot infer baseline type from {bpath.name}, assuming cno (use --baseline-model)")
                    btype = "cno"
            from load_baseline import load_baseline
            print(f"[eval] loading baseline {btype} from {bpath} ...")
            baseline, meta = load_baseline(btype, str(bpath), device=args.device)
            baseline.eval()
            print(f"[eval] baseline loaded: {btype} {meta}")

    # Eval (raw-space MSE + rel-L2; normalized MSE for ckpt val_loss comparison)
    def score_model(m):
        mse_sum = 0.0
        n_elem = 0
        rel_sum = 0.0
        nmse_sum = 0.0
        n = 0
        with torch.no_grad():
            for inp, tgt in loader:
                inp_raw, tgt_raw = inp.to(args.device), tgt.to(args.device)
                inp_n, tgt_n = norm.preprocess(inp_raw, tgt_raw)
                pred_n = m(inp_n)
                pred_raw = norm.postprocess_pred(pred_n)
                nmse_sum += F.mse_loss(pred_n, tgt_n, reduction="sum").item()
                mse_sum += F.mse_loss(pred_raw, tgt_raw, reduction="sum").item()
                n_elem += tgt_raw.numel()
                rel_sum += rel_l2_batch(pred_raw, tgt_raw) * inp_raw.shape[0]
                n += inp_raw.shape[0]
        mse = mse_sum / max(n_elem, 1)
        rel = rel_sum / max(n, 1)
        nmse = nmse_sum / max(n_elem, 1)
        return mse, rel, nmse, n

    def score_baseline(m):
        # Shipped baselines were trained with the official stats; score raw.
        mse_sum = 0.0
        n_elem = 0
        rel_sum = 0.0
        n = 0
        with torch.no_grad():
            for inp, tgt in loader:
                inp_raw, tgt_raw = inp.to(args.device), tgt.to(args.device)
                pred_raw = m(inp_raw)
                mse_sum += F.mse_loss(pred_raw, tgt_raw, reduction="sum").item()
                n_elem += tgt_raw.numel()
                rel_sum += rel_l2_batch(pred_raw, tgt_raw) * inp_raw.shape[0]
                n += inp_raw.shape[0]
        return mse_sum / max(n_elem, 1), rel_sum / max(n, 1), n

    mse, rel, nmse, n = score_model(model)
    print(f"\n[eval] UNet  | n={n} | MSE {mse:.6f} | rel-L2 {rel:.6f} | norm-MSE {nmse:.6f} (compare to ckpt val_loss)")

    if baseline is not None:
        mse_b, rel_b, n_b = score_baseline(baseline)
        print(f"[eval] {args.baseline_model or btype:5s} | n={n_b} | MSE {mse_b:.6f} | rel-L2 {rel_b:.6f}")
        if rel > 0:
            print(f"[eval] delta vs baseline: rel-L2 UNet/baseline = {rel/rel_b:.3f}x  ( <1 means yours is better )")

    # Also print ckpt val_loss if present
    if isinstance(obj, dict) and "val_loss" in obj:
        print(f"[eval] ckpt val_loss (from trainer): {obj['val_loss']:.6f}  epoch {obj.get('epoch','?')}")

if __name__ == "__main__":
    main()
