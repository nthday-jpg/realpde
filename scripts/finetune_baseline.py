#!/usr/bin/env python3
"""Finetune a shipped baseline (CNO/FNO/Transolver) with HF accelerate.

Notebook calls it via:
    accelerate launch --mixed_precision=fp16 --num_processes=<n_gpus> scripts/finetune_baseline.py

Goal of the current study: check whether the shipped sim CNO
(data/baseline_checkpoints/sim_pretrain/sim_cno.pth) is undertrained, by
resuming it on train_sim (same distribution) and on train_real (finetune)
and comparing val curves.

Config via env (set in notebook CONFIG cell):
    DATA_PATH       dir with .h5 files (train_sim or train_real)
    DATA_CACHE      optional .pt from scripts/cache_dataset.py (skips .h5 reads)
    RESUME_CKPT     path to .pth to resume (shipped ckpt or prior run)
    MODEL_TYPE      auto-detected from filename if unset (cno/fno/transolver)
    IN_STEP/OUT_STEP/INTERVAL/SUB_S (default 20/20/20/2)
    LR (1e-4), EPOCHS (20), BATCH_SIZE (4), WEIGHT_DECAY (0)
    VAL_FRAC (0.1 of FILES), SEED (42, trajectory split), NUM_WORKERS (unset=auto)
    REAL_DATA_PATH  dir with real .h5 for post-train test (unset = skip)
    REAL_FRAC (0.2 of real files), REAL_SEED (7), EVAL_BATCH_SIZE (=BATCH_SIZE)
    SAVE_DIR (/kaggle/working/cno_continue)
    WANDB_PROJECT (realpde-finetune), WANDB_RUN_NAME (optional)

Checkpoint format matches load_baseline: {'model_state_dict', ...} plus
optimizer_state_dict/epoch/val_loss, so baseline_kaggle.ipynb can score them
and reruns chain (optimizer/epoch restored when present).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)  # scripts/ -> repo root
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realpde.datasets import (
    PDEDataset,
    file_window_counts,
    trajectory_split_indices,
)
from load_baseline import load_baseline, detect_model_type


def _cfg(key, default=None):
    return os.environ.get(key, default)


def _test_on_real(real_data, continued_ckpt, before_ckpt, model_type,
                 in_step, out_step, interval, sub_s, batch_size,
                 save_dir, accelerator):
    """Score continued-best vs before-train ckpt on REAL_FRAC of real files.

    Reports raw-space MSE + rel-L2 (same formula as baseline_kaggle.ipynb).
    """
    import gc
    import random as _random
    import shutil as _shutil
    from collections import Counter as _Counter
    from pathlib import Path as _Path
    from torch.utils.data import DataLoader as _DataLoader

    real_frac = float(os.environ.get("REAL_FRAC", 0.2))
    real_seed = int(os.environ.get("REAL_SEED", 7))
    eval_batch = int(os.environ.get("EVAL_BATCH_SIZE", batch_size))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = _Path(real_data)
    direct = sorted(root.glob("*.h5"))
    h5dir = root if direct else _Counter(
        p.parent for p in root.rglob("*.h5")).most_common(1)[0][0]
    files = sorted(h5dir.glob("*.h5"))
    files = list(files)
    _random.Random(real_seed).shuffle(files)
    sel = files[:max(1, int(len(files) * real_frac))]
    print(f"[RealTest] {len(sel)}/{len(files)} real files from {h5dir} "
          f"(frac={real_frac} seed={real_seed})")

    stage = _Path(save_dir) / "real_eval_stage"
    if stage.exists():
        _shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in sel:
        (stage / f.name).symlink_to(f)
    eval_ds = PDEDataset(stage, in_step=in_step, out_step=out_step,
                         interval=interval, sub_s=sub_s)
    loader = _DataLoader(eval_ds, batch_size=eval_batch, shuffle=False)
    print(f"[RealTest] {len(eval_ds)} windows, batch={eval_batch}")

    accelerator.free_memory()  # drop train model/opt before loading eval models
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _score(ckpt_path, tag):
        m, _ = load_baseline(model_type, str(ckpt_path), device=device)
        m.eval()
        ms, rs, n = 0.0, 0.0, 0
        with torch.no_grad():
            for inp, tgt in loader:
                inp, tgt = inp.to(device), tgt.to(device)
                pred = m(inp)
                ms += F.mse_loss(pred, tgt, reduction="sum").item()
                b = pred.shape[0]
                p, t = pred.reshape(b, -1), tgt.reshape(b, -1)
                rs += ((p - t).norm(dim=1)
                       / t.norm(dim=1).clamp_min(1e-8)).sum().item()
                n += b
        del m
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mse = ms / max(n * out_step * 32 * 64 * 3, 1)
        rel = rs / max(n, 1)
        print(f"[RealTest] {tag:16s} MSE {mse:.6f}  rel-L2 {rel:.6f}  (n={n})")
        return mse, rel

    mse1, rel1 = _score(continued_ckpt, "continued-best")
    mse0, rel0 = _score(before_ckpt, "before-train")
    print(f"[RealTest] gain (before/continued): "
          f"MSE {mse0 / max(mse1, 1e-12):.2f}x  rel-L2 {rel0 / max(rel1, 1e-12):.2f}x")


def main():
    # CNO3d always leaves one decoder_inv block out of the forward graph
    # (loop uses indices 0..N_layers-1 of an N_layers+1 ModuleList when
    # add_inv=True), so DDP needs find_unused_parameters=True. Single-GPU is
    # unaffected.
    from accelerate.utils import DistributedDataParallelKwargs
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        log_with="wandb",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    data_path = str(_cfg("DATA_PATH", "data/train_sim"))
    resume = str(_cfg("RESUME_CKPT", "data/baseline_checkpoints/sim_pretrain/sim_cno.pth"))
    model_type = _cfg("MODEL_TYPE") or detect_model_type(resume)
    in_step = int(_cfg("IN_STEP", 20))
    out_step = int(_cfg("OUT_STEP", 20))
    interval = int(_cfg("INTERVAL", 20))
    sub_s = int(_cfg("SUB_S", 2))
    lr = float(_cfg("LR", 1e-4))
    epochs = int(_cfg("EPOCHS", 20))
    batch_size = int(_cfg("BATCH_SIZE", 4))
    wd = float(_cfg("WEIGHT_DECAY", 0))
    val_frac = float(_cfg("VAL_FRAC", 0.1))
    seed = int(_cfg("SEED", 42))
    save_dir = str(_cfg("SAVE_DIR", "/kaggle/working/cno_continue"))
    wandb_project = str(_cfg("WANDB_PROJECT", "realpde-finetune"))
    wandb_run = _cfg("WANDB_RUN_NAME", None)

    data_cache = _cfg("DATA_CACHE", "")
    if data_cache and os.path.exists(data_cache):
        # Fast path: windows cached by scripts/cache_dataset.py (torch.load
        # runs on each rank; file is on local disk so this is quick).
        blob = torch.load(data_cache, map_location="cpu", weights_only=False)
        full = torch.utils.data.TensorDataset(blob["inputs"], blob["targets"])
        # file_counts saved by cache_dataset; older caches predate it -> recompute
        # from DATA_PATH (cheap shape-only reads).
        counts = ([tuple(c) for c in blob["file_counts"]] if "file_counts" in blob
                  else file_window_counts(data_path, in_step, out_step, interval))
        if accelerator.is_main_process:
            print(f"[Data] loaded cache {data_cache}: {blob['n']} windows "
                  f"(from {blob.get('data_path')})")
    else:
        full = PDEDataset(data_path, in_step=in_step, out_step=out_step,
                          interval=interval, sub_s=sub_s)
        counts = file_window_counts(data_path, in_step, out_step, interval)
    # Split by TRAJECTORY/file: val files are fully unseen trajectories.
    assert sum(n for _, n in counts) == len(full), \
        "file counts disagree with dataset length (re-cache if knobs changed)"
    train_idx, val_idx = trajectory_split_indices(counts, val_frac, seed)
    train_ds, val_ds = Subset(full, train_idx), Subset(full, val_idx)
    if accelerator.is_main_process:
        print(f"[Split] {len(counts)} files -> train {len(train_idx)} windows / "
              f"val {len(val_idx)} windows (by trajectory, seed={seed})")

    nw_env = os.environ.get("NUM_WORKERS", "").strip()
    if nw_env:
        num_workers = int(nw_env)
    else:
        cpu = os.cpu_count() or 4
        try:
            n_proc = int(accelerator.num_processes) or 1
        except Exception:
            n_proc = 1
        num_workers = min(max(1, cpu // max(n_proc, 1)), 8)
    if accelerator.is_main_process:
        print(f"[Data] {data_path}: train={len(train_ds)} val={len(val_ds)} "
              f"workers={num_workers} x{accelerator.num_processes}")

    pin = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)

    # Build arch on CPU via load_baseline, then accelerate prepares it.
    # load_baseline puts model on `device`; pass cpu here, accelerator moves it.
    model, meta = load_baseline(model_type, resume, device="cpu")
    if accelerator.is_main_process:
        print(f"[Model] {model_type} resumed from {resume} meta={meta}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    start_epoch, best_val = 0, float("inf")
    raw = torch.load(resume, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "optimizer_state_dict" in raw:
        try:
            opt.load_state_dict(raw["optimizer_state_dict"])
            start_epoch = int(raw.get("epoch", -1)) + 1
            best_val = float(raw.get("val_loss", float("inf")))
            if accelerator.is_main_process:
                print(f"[Resume] chained: start_epoch={start_epoch} best_val={best_val:.6f}")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Resume] optimizer restore skipped: {e}")

    model, opt, train_loader, val_loader = accelerator.prepare(
        model, opt, train_loader, val_loader)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=dict(data_path=data_path, resume=resume, model_type=model_type,
                        in_step=in_step, out_step=out_step, interval=interval,
                        sub_s=sub_s, lr=lr, epochs=epochs, batch_size=batch_size,
                        seed=seed),
            init_kwargs={"wandb": {"name": wandb_run}} if wandb_run else {},
        )
        os.makedirs(save_dir, exist_ok=True)

    end_epoch = start_epoch + epochs
    global_step = 0
    for epoch in range(start_epoch, end_epoch):
        model.train()
        tot = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}",
                    disable=not accelerator.is_main_process, leave=False)
        for inp, tgt in pbar:
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(inp), tgt)
            accelerator.backward(loss)
            opt.step()
            tot += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            if global_step % 50 == 0 and accelerator.is_main_process:
                accelerator.log({"train_loss": loss.item(), "epoch": epoch},
                                step=global_step)
        train_loss = tot / max(len(train_loader), 1)

        model.eval()
        vt = 0.0
        with torch.no_grad():
            for inp, tgt in val_loader:
                vt += F.mse_loss(model(inp), tgt).item()
        # average across processes
        vt_t = torch.tensor([vt, len(val_loader)], device=accelerator.device)
        accelerator.reduce(vt_t, reduction="sum")
        val_loss = (vt_t[0] / max(int(vt_t[1].item()), 1)).item()

        if accelerator.is_main_process:
            accelerator.log({"val_loss": val_loss, "train_loss_epoch": train_loss,
                             "epoch": epoch}, step=global_step)
            tag = "  <-- best" if val_loss < best_val else ""
            print(f"Epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f}{tag}")
            unwrapped = accelerator.unwrap_model(model)
            state = {"model_state_dict": {k: v.cpu() for k, v in unwrapped.state_dict().items()},
                     "optimizer_state_dict": opt.state_dict(),
                     "epoch": epoch, "val_loss": val_loss, "train_loss": train_loss,
                     "cfg": dict(in_step=in_step, out_step=out_step, interval=interval,
                                 sub_s=sub_s, lr=lr, model_type=model_type),
                     "resume_from": resume}
            torch.save(state, os.path.join(save_dir, "last.pth"))
            if val_loss < best_val:
                best_val = val_loss
                torch.save(state, os.path.join(save_dir, "best.pth"))
            if (epoch + 1) % 5 == 0:
                torch.save(state, os.path.join(save_dir, f"epoch_{epoch:03d}.pth"))

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        state = {"model_state_dict": {k: v.cpu() for k, v in unwrapped.state_dict().items()},
                 "optimizer_state_dict": opt.state_dict(),
                 "epoch": end_epoch - 1, "val_loss": val_loss,
                 "cfg": dict(model_type=model_type), "resume_from": resume}
        torch.save(state, os.path.join(save_dir, "final.pth"))
        print(f"Done. best val {best_val:.6f} -> {save_dir}/best.pth")

        # --- Post-train test: continued model vs before-train ckpt on real ---
        # Main process only. Frees the training model first so the two eval
        # models load one at a time (no OOM). Configure with REAL_DATA_PATH
        # (dir with real .h5) + REAL_FRAC (default 0.2 of files, seeded).
        real_data = _cfg("REAL_DATA_PATH", "")
        if real_data and os.path.isdir(real_data):
            _test_on_real(real_data, os.path.join(save_dir, "best.pth"),
                           resume, model_type, in_step, out_step, interval,
                           sub_s, batch_size, save_dir, accelerator)
        elif real_data:
            print(f"[RealTest] skip: not a dir: {real_data}")
        else:
            print("[RealTest] skip: set REAL_DATA_PATH to enable")
        accelerator.end_training()


if __name__ == "__main__":
    main()
