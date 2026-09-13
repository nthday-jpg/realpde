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
    RESUME_CKPT     path to .pth to resume (shipped ckpt or prior run)
    MODEL_TYPE      auto-detected from filename if unset (cno/fno/transolver)
    IN_STEP/OUT_STEP/INTERVAL/SUB_S (default 20/20/20/2)
    LR (1e-4), EPOCHS (20), BATCH_SIZE (4), WEIGHT_DECAY (0)
    VAL_FRAC (0.1), SEED (42), NUM_WORKERS (unset=auto)
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
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)  # scripts/ -> repo root
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realpde.datasets import PDEDataset
from load_baseline import load_baseline, detect_model_type


def _cfg(key, default=None):
    return os.environ.get(key, default)


def main():
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        log_with="wandb",
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

    g = torch.Generator().manual_seed(seed)
    full = PDEDataset(data_path, in_step=in_step, out_step=out_step,
                      interval=interval, sub_s=sub_s)
    n_val = max(1, int(val_frac * len(full)))
    train_ds, val_ds = random_split(full, [len(full) - n_val, n_val], generator=g)

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
        accelerator.end_training()
        print(f"Done. best val {best_val:.6f} -> {save_dir}/best.pth")


if __name__ == "__main__":
    main()
