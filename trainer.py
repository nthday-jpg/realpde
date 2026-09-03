#!/usr/bin/env python3
"""Pretraining script for PDE velocity field models.

Launch with ``accelerate launch trainer.py`` (config set in notebook globals
before launch).  Supports teacher-forcing pretraining with configurable
in/out step windows and multiple model architectures.

Config variables (set in notebook or environment before launch):
    DATA_PATH       : str   – path to directory with .h5 files
    MODEL_NAME      : str   – model name (e.g. "unet")
    IN_STEP         : int   – input frames (default 20)
    OUT_STEP        : int   – output frames (default 20)
    INTERVAL        : int   – temporal stride (default 20)
    LR              : float – learning rate (default 1e-3)
    EPOCHS          : int   – number of epochs (default 50)
    BATCH_SIZE      : int   – local batch size (default 8)
    SUB_S           : int   – spatial subsample factor (default 2)
    SAVE_DIR        : str   – checkpoint output dir (default "checkpoints")
    WANDB_PROJECT   : str   – wandb project name (default "realpde-pretrain")
    WANDB_RUN_NAME  : str   – wandb run name (default auto)
    CHECKPOINT_PATH : str   – path to resume from (default None)
    UNET_CHANNELS   : int   – UNet base channels (default 16)
    UNET_N_LAYERS   : int   – UNet encoder depth (default 2)
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
_SRC = os.path.join(_HERE, "src")
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realpde.datasets import PDEDataset
from realpde.models import get_model, ModelAdapter


def _cfg(key: str, default=None):
    """Read a config value from globals (notebook-set) or env."""
    return globals().get(key, os.environ.get(key, default))


def main():
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        log_with="wandb",
    )

    # --- Config ----------------------------------------------------------------
    data_path = str(_cfg("DATA_PATH", "data/train_sim"))
    model_name = str(_cfg("MODEL_NAME", "unet"))
    in_step = int(_cfg("IN_STEP", 20))
    out_step = int(_cfg("OUT_STEP", 20))
    interval = int(_cfg("INTERVAL", 20))
    lr = float(_cfg("LR", 1e-3))
    epochs = int(_cfg("EPOCHS", 50))
    batch_size = int(_cfg("BATCH_SIZE", 8))
    sub_s = int(_cfg("SUB_S", 2))
    save_dir = str(_cfg("SAVE_DIR", "checkpoints"))
    wandb_project = str(_cfg("WANDB_PROJECT", "realpde-pretrain"))
    wandb_run_name = _cfg("WANDB_RUN_NAME", None)
    checkpoint_path = _cfg("CHECKPOINT_PATH", None)
    unet_channels = int(_cfg("UNET_CHANNELS", 16))
    unet_n_layers = int(_cfg("UNET_N_LAYERS", 2))

    cfg = dict(
        data_path=data_path, model_name=model_name, in_step=in_step,
        out_step=out_step, interval=interval, lr=lr, epochs=epochs,
        batch_size=batch_size, sub_s=sub_s, save_dir=save_dir,
        unet_channels=unet_channels, unet_n_layers=unet_n_layers,
    )

    # --- Model -----------------------------------------------------------------
    if model_name == "unet":
        from realpde.models.unet import UNet, UNetConfig
        model_cfg = UNetConfig(
            in_step=in_step, out_step=out_step, channels=unet_channels,
            n_layers=unet_n_layers,
        )
        model = UNet(model_cfg)
    else:
        model = get_model(model_name)

    adapter = ModelAdapter.from_pretrained(model, checkpoint_path)
    base = adapter.model

    optimizer = torch.optim.Adam(base.parameters(), lr=lr)

    # --- Data ------------------------------------------------------------------
    full_dataset = PDEDataset(data_path, in_step=in_step, out_step=out_step,
                              interval=interval, sub_s=sub_s)
    n_val = max(1, int(0.1 * len(full_dataset)))
    train_ds, val_ds = random_split(full_dataset, [len(full_dataset) - n_val, n_val])

    # --- DataLoader workers: use maximum sensible, allow override via NUM_WORKERS ---
    _num_workers_env = os.environ.get("NUM_WORKERS", "").strip()
    if _num_workers_env != "":
        try:
            num_workers = int(_num_workers_env)
        except ValueError:
            num_workers = 0
    else:
        # auto: all CPUs split across processes, capped to 8 to avoid oversubscription
        cpu_count = os.cpu_count() or 4
        try:
            n_proc = int(accelerator.num_processes) or 1
        except Exception:
            n_proc = 1
        if n_proc > 1:
            # per-process workers so total ≈ cpu_count
            num_workers = max(1, cpu_count // n_proc)
        else:
            num_workers = cpu_count
        num_workers = min(num_workers, 8)
        # Kaggle T4x2: 4 CPUs -> 2 workers per rank; high-CPU machines -> up to 8
    # accelerate handles multi-GPU sharding; DataLoader itself stays simple
    _pin = torch.cuda.is_available()
    _persistent = num_workers > 0
    _prefetch = 2 if num_workers > 0 else None
    if accelerator.is_main_process:
        print(f"[Data] cpu_count={os.cpu_count()} num_processes={accelerator.num_processes} -> num_workers={num_workers} (override with NUM_WORKERS env)")

    _loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=_pin,
        persistent_workers=_persistent,
    )
    if _prefetch is not None:
        _loader_kwargs["prefetch_factor"] = _prefetch
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **_loader_kwargs)

    base, optimizer, train_loader, val_loader = accelerator.prepare(
        base, optimizer, train_loader, val_loader
    )

    # --- Logging ---------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=cfg,
            init_kwargs={"wandb": {"name": wandb_run_name}} if wandb_run_name else {},
        )

    # --- Training loop ---------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        base.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", disable=not accelerator.is_main_process)
        for batch_idx, (inp, tgt) in enumerate(pbar):
            optimizer.zero_grad()
            pred = base(inp)
            loss = F.mse_loss(pred, tgt)
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

            if global_step % 50 == 0 and accelerator.is_main_process:
                accelerator.log({"train_loss": loss.item(), "epoch": epoch, "lr": lr},
                                step=global_step)

        avg_train = train_loss / max(len(train_loader), 1)

        # --- Validation --------------------------------------------------------
        base.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc="  Val  ", disable=not accelerator.is_main_process)
        with torch.no_grad():
            for inp, tgt in val_pbar:
                pred = base(inp)
                batch_val = F.mse_loss(pred, tgt).item()
                val_loss += batch_val
                val_pbar.set_postfix(loss=f"{batch_val:.6f}")
        avg_val = val_loss / max(len(val_loader), 1)

        if accelerator.is_main_process:
            accelerator.log({"val_loss": avg_val, "train_loss_epoch": avg_train,
                             "epoch": epoch}, step=global_step)
            print(f"Epoch {epoch:3d} | train {avg_train:.6f} | val {avg_val:.6f}")

            # Save best checkpoint
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                unwrapped = accelerator.unwrap_model(base)
                state = {"model_state_dict": unwrapped.state_dict(),
                         "epoch": epoch, "val_loss": avg_val, "cfg": cfg}
                torch.save(state, os.path.join(save_dir, "best.pth"))

            # Save latest checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                unwrapped = accelerator.unwrap_model(base)
                state = {"model_state_dict": unwrapped.state_dict(),
                         "epoch": epoch, "val_loss": avg_val, "cfg": cfg}
                torch.save(state, os.path.join(save_dir, f"epoch_{epoch:03d}.pth"))

    # --- Final save ------------------------------------------------------------
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(base)
        state = {"model_state_dict": unwrapped.state_dict(),
                 "epoch": epochs - 1, "val_loss": avg_val, "cfg": cfg}
        torch.save(state, os.path.join(save_dir, "final.pth"))
        accelerator.end_training()
        print(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
