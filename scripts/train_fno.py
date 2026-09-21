#!/usr/bin/env python3
"""Train a configurable FNO3d with Hugging Face Accelerate.

The companion ``notebook/train_fno_kaggle.ipynb`` owns the user-facing
configuration and launches this script with ``accelerate launch``. All options
must be supplied as environment variables; see ``Config.from_env``.
"""
from __future__ import annotations

import math
import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parents[1]
for _path in (str(_REPO / "src"), str(_REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from realpde.datasets import (  # noqa: E402
    PDEDataset,
    PDENormalizer,
    file_window_counts,
    trajectory_split_indices,
)
from realpde.rpde_baselines.model.fno import FNO3d, SpectralConv3d  # noqa: E402


def _env(name: str) -> str:
    return os.environ[name]


def _bool_env(name: str) -> bool:
    value = _env(name).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass
class Config:
    data_path: str
    data_cache: str
    save_dir: str
    resume_ckpt: str
    in_step: int
    out_step: int
    interval: int
    sub_s: int
    val_frac: float
    seed: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    grad_accum_steps: int
    max_grad_norm: float
    mixed_precision: str
    modes1: int
    modes2: int
    modes3: int
    n_layers: int
    width: int
    padding: int
    save_every: int
    save_optimizer: bool
    log_every_steps: int
    wandb_project: str
    wandb_run_name: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            data_path=_env("DATA_PATH"),
            data_cache=_env("DATA_CACHE"),
            save_dir=_env("SAVE_DIR"),
            resume_ckpt=_env("RESUME_CKPT"),
            in_step=int(_env("IN_STEP")),
            out_step=int(_env("OUT_STEP")),
            interval=int(_env("INTERVAL")),
            sub_s=int(_env("SUB_S")),
            val_frac=float(_env("VAL_FRAC")),
            seed=int(_env("SEED")),
            batch_size=int(_env("BATCH_SIZE")),
            num_workers=int(_env("NUM_WORKERS")),
            epochs=int(_env("EPOCHS")),
            lr=float(_env("LR")),
            weight_decay=float(_env("WEIGHT_DECAY")),
            grad_accum_steps=int(_env("GRAD_ACCUM_STEPS")),
            max_grad_norm=float(_env("MAX_GRAD_NORM")),
            mixed_precision=_env("MIXED_PRECISION"),
            modes1=int(_env("FNO_MODES1")),
            modes2=int(_env("FNO_MODES2")),
            modes3=int(_env("FNO_MODES3")),
            n_layers=int(_env("FNO_N_LAYERS")),
            width=int(_env("FNO_WIDTH")),
            padding=int(_env("FNO_PADDING")),
            save_every=int(_env("SAVE_EVERY")),
            save_optimizer=_bool_env("SAVE_OPTIMIZER"),
            log_every_steps=int(_env("LOG_EVERY_STEPS")),
            wandb_project=_env("WANDB_PROJECT"),
            wandb_run_name=_env("WANDB_RUN_NAME"),
        )


def _load_data(cfg: Config, accelerator: Accelerator):
    cache = Path(cfg.data_cache) if cfg.data_cache else None
    if cache and cache.is_file():
        # mmap keeps the multi-GB tensor storage file-backed. DDP ranks then
        # share the OS page cache instead of each allocating a full RAM copy.
        try:
            blob = torch.load(
                cache, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:  # compatibility with older PyTorch releases
            blob = torch.load(cache, map_location="cpu", weights_only=False)
        full = TensorDataset(blob["inputs"], blob["targets"])
        counts = ([tuple(item) for item in blob["file_counts"]]
                  if "file_counts" in blob else
                  file_window_counts(cfg.data_path, cfg.in_step, cfg.out_step, cfg.interval))
        accelerator.print(f"[Data] loaded {len(full)} windows from cache {cache}")
    else:
        blob = None
        full = PDEDataset(
            cfg.data_path,
            in_step=cfg.in_step,
            out_step=cfg.out_step,
            interval=cfg.interval,
            sub_s=cfg.sub_s,
        )
        counts = file_window_counts(cfg.data_path, cfg.in_step, cfg.out_step, cfg.interval)
        accelerator.print(f"[Data] loaded {len(full)} windows from {cfg.data_path}")

    if sum(n for _, n in counts) != len(full):
        raise ValueError("Dataset window count mismatch; rebuild DATA_CACHE with the current window settings")
    train_idx, val_idx = trajectory_split_indices(counts, cfg.val_frac, cfg.seed)
    if not train_idx or not val_idx:
        raise ValueError("The trajectory split produced an empty train or validation set")

    # Never use blob["inputs"][train_idx] here: advanced indexing materializes
    # multi-GB copies, and fit_from_samples then converts them to float64. That
    # can exceed Kaggle RAM, especially when every DDP rank loads the cache.
    # Stream one window at a time on rank 0 and broadcast the four tiny stats
    # tensors to the other ranks instead.
    stats = [None, None]
    if accelerator.is_main_process:
        accelerator.print("[Norm] fitting streaming statistics on rank 0 ...")
        train_norm_main = PDENormalizer.fit_from_indexed(full, train_idx)
        val_norm_main = PDENormalizer.fit_from_indexed(full, val_idx)
        stats = [train_norm_main.as_tuple(), val_norm_main.as_tuple()]
    stats = broadcast_object_list(stats, from_process=0)
    train_norm = PDENormalizer(*stats[0])
    val_norm = PDENormalizer(*stats[1])

    common = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(full, train_idx), batch_size=cfg.batch_size, shuffle=True,
        drop_last=False, **common)
    val_loader = DataLoader(
        Subset(full, val_idx), batch_size=cfg.batch_size, shuffle=False,
        drop_last=False, **common)
    accelerator.print(
        f"[Split] {len(train_idx)} train / {len(val_idx)} val windows; "
        f"train norm: {train_norm.describe()}")
    return full, train_loader, val_loader, train_norm, val_norm


def _state_dict_from_checkpoint(obj):
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_fp16" in obj:
        complex_keys = set(obj.get("complex_keys", []))
        return {
            key: (torch.view_as_complex(value.float()) if key in complex_keys
                  else value.float() if torch.is_tensor(value) and value.dtype == torch.float16
                  else value)
            for key, value in obj["state_fp16"].items()
        }
    return obj


def _amp_safe_spectral_forward(self, x):
    """Training-only FP32 spectral path without modifying committee code.

    CUDA half-precision FFT requires power-of-two dimensions, while baseline
    padding produces (26, 38, 70). Pointwise layers remain under AMP; only FFT
    and complex multiplication run in fp32/complex64. State-dict keys and the
    baseline architecture are unchanged.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-3), x.size(-2),
            x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        return torch.fft.irfftn(
            out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


def _build_model(cfg: Config, sample):
    input_shape, output_shape = tuple(sample[0].shape), tuple(sample[1].shape)
    if output_shape[0] % input_shape[0] != 0:
        raise ValueError("FNO3d requires OUT_STEP to be an integer multiple of IN_STEP")
    if input_shape[1:3] != output_shape[1:3]:
        raise ValueError("FNO3d requires matching input/output spatial shapes")
    padded = (input_shape[0] + cfg.padding,
              input_shape[1] + cfg.padding,
              input_shape[2] + cfg.padding)
    limits = (padded[0], padded[1], padded[2] // 2 + 1)
    modes = (cfg.modes1, cfg.modes2, cfg.modes3)
    if any(mode <= 0 or mode > limit for mode, limit in zip(modes, limits)):
        raise ValueError(f"FNO modes {modes} exceed padded FFT limits {limits}")

    model = FNO3d(
        modes1=cfg.modes1,
        modes2=cfg.modes2,
        modes3=cfg.modes3,
        n_layers=cfg.n_layers,
        width=cfg.width,
        shape_in=input_shape,
        shape_out=output_shape,
    )
    model.padding = cfg.padding
    if cfg.mixed_precision != "no":
        for module in model.modules():
            if isinstance(module, SpectralConv3d):
                module.forward = types.MethodType(_amp_safe_spectral_forward, module)
    return model


def _scoring_metric_totals(pred: torch.Tensor, target: torch.Tensor,
                           sub_s_real: int = 2) -> torch.Tensor:
    """Return additive raw-space validation totals matching ``scoring.py``.

    The dataset always has measured ``u,v`` plus a zero-filled ``p`` channel,
    so scoring uses the first two channels. SPS uses the scorer's fallback
    interval (prediction +/- 5% of its absolute value).
    """
    c = min(2, pred.shape[-1])
    p, t = pred[..., :c], target[..., :c]
    batch = pred.shape[0]

    p_flat, t_flat = p.reshape(batch, -1), t.reshape(batch, -1)
    dm = (p_flat - t_flat).norm(dim=1) / t_flat.norm(dim=1).clamp_min(1e-8)

    if c >= 2:
        def kinetic_energy(x):
            u, v = x[..., 0], x[..., 1]
            u_prime = ((u - u.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
            v_prime = ((v - v.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
            return 0.5 * (u_prime + v_prime)

        pred_ke, target_ke = kinetic_energy(p), kinetic_energy(t)
        tke = ((pred_ke - target_ke).reshape(batch, -1).norm(dim=1)
               / target_ke.reshape(batch, -1).norm(dim=1).clamp_min(1e-8))
    else:
        tke = torch.zeros_like(dm)

    # Mean velocity profile error at the same probes as scoring.py.
    d, center_x, center_y, n_probe = 16, 10, 32, 9
    h, w = pred.shape[2], pred.shape[3]
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(h / (n_probe + 1)))
    probe_y = [
        probe_center_y + interval_y * j
        for j in range(-(n_probe - 1) // 2,
                       n_probe - (n_probe - 1) // 2)
    ]
    probe_y = [y for y in probe_y if 0 <= y < h]
    mvpe_parts = []
    if c >= 2 and probe_y:
        for i in range(4):
            if int((2 * d + center_x) / sub_s_real) < w:
                probe_x = int(((i + 1) * d + center_x) / sub_s_real)
            else:
                probe_x = int((0.5 * (i + 2) * d + center_x) / sub_s_real)
            if not 0 <= probe_x < w:
                continue
            pp = p[:, :, probe_y, probe_x, :2].mean(dim=1).reshape(batch, -1)
            tt = t[:, :, probe_y, probe_x, :2].mean(dim=1).reshape(batch, -1)
            mvpe_parts.append(
                (pp - tt).norm(dim=1) / tt.norm(dim=1).clamp_min(1e-8))
    mvpe = (torch.stack(mvpe_parts).mean(dim=0) if mvpe_parts
            else torch.zeros_like(dm))

    # SPS fallback bounds and branch aggregation from scoring.py.
    lower = p - 0.05 * p.abs()
    upper = p + 0.05 * p.abs()
    inside = (t >= lower) & (t <= upper)
    nil = (upper - lower) / 0.0563870259

    def sps_branch(error):
        pm = error / (0.5 + error)
        shaped = pm.reshape((batch,) + (1,) * (p.ndim - 1))
        elem = (1.0 - shaped) * torch.exp(-nil)
        return torch.where(inside, elem, torch.zeros_like(elem)).sum()

    return torch.stack([
        ((p - t) ** 2).sum(),
        p.new_tensor(p.numel()),
        dm.sum(),
        tke.sum(),
        mvpe.sum(),
        p.new_tensor(batch),
        sps_branch(dm),
        sps_branch(tke),
        sps_branch(mvpe),
        inside.sum().to(dtype=p.dtype),
        p.new_tensor(p.numel()),
    ])


def _error_score(error: float) -> float:
    return 100.0 / (1.0 + 0.5 * max(error, 0.0))


def _checkpoint(cfg, accelerator, model, optimizer, scheduler, epoch, global_step,
                train_loss, val_loss, train_norm, val_norm):
    state = {
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in accelerator.get_state_dict(model).items()
        },
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "cfg": asdict(cfg),
        "norm_train": train_norm.as_tuple(),
        "norm_val": val_norm.as_tuple(),
    }
    if cfg.save_optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["scheduler_state_dict"] = scheduler.state_dict()
    return state


def main() -> None:
    cfg = Config.from_env()
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        log_with="wandb" if cfg.wandb_project else None,
    )
    if accelerator.mixed_precision != cfg.mixed_precision:
        raise RuntimeError(
            f"Precision mismatch: config requested {cfg.mixed_precision!r}, "
            f"but Accelerate activated {accelerator.mixed_precision!r}"
        )
    accelerator.print(
        f"[Precision] requested={cfg.mixed_precision} "
        f"active={accelerator.mixed_precision}")
    torch.manual_seed(cfg.seed)

    full, train_loader, val_loader, train_norm, val_norm = _load_data(cfg, accelerator)
    model = _build_model(cfg, full[0])
    accelerator.print(
        f"[Model] FNO3d modes=({cfg.modes1},{cfg.modes2},{cfg.modes3}) "
        f"layers={cfg.n_layers} width={cfg.width} padding={cfg.padding}; "
        f"{sum(p.numel() for p in model.parameters()):,} parameter elements")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_epoch, global_step, best_val = 0, 0, math.inf
    resume_obj = None
    if cfg.resume_ckpt:
        resume_obj = torch.load(cfg.resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(_state_dict_from_checkpoint(resume_obj), strict=True)
        if isinstance(resume_obj, dict):
            start_epoch = int(resume_obj.get("epoch", -1)) + 1
            global_step = int(resume_obj.get("global_step", 0))
            best_val = float(resume_obj.get("val_loss", math.inf))
        accelerator.print(f"[Resume] model loaded from {cfg.resume_ckpt}")

    updates_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_updates = max(cfg.epochs * updates_per_epoch, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=cfg.lr * 0.01)
    if isinstance(resume_obj, dict) and "optimizer_state_dict" in resume_obj:
        optimizer.load_state_dict(resume_obj["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_obj:
            scheduler.load_state_dict(resume_obj["scheduler_state_dict"])
        accelerator.print(f"[Resume] optimizer restored; starting at epoch {start_epoch}")

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)

    save_dir = Path(cfg.save_dir)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        train_norm.save(save_dir / "mean_std_train.pt")
        val_norm.save(save_dir / "mean_std_val.pt")
        if cfg.wandb_project:
            accelerator.init_trackers(
                cfg.wandb_project,
                config=asdict(cfg),
                init_kwargs={"wandb": {"name": cfg.wandb_run_name}}
                if cfg.wandb_run_name else {},
            )
    accelerator.wait_for_everyone()

    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        model.train()
        train_loss_sum = torch.zeros((), device=accelerator.device)
        train_elements = torch.zeros((), device=accelerator.device)
        progress = tqdm(
            train_loader,
            disable=not accelerator.is_local_main_process,
            desc=f"Train {epoch:03d}",
            leave=True,
            dynamic_ncols=True,
        )
        optimizer.zero_grad(set_to_none=True)
        for inputs, targets in progress:
            inputs, targets = train_norm.preprocess(inputs, targets)
            with accelerator.accumulate(model):
                predictions = model(inputs)
                loss = F.mse_loss(predictions, targets)
                accelerator.backward(loss)
                if accelerator.sync_gradients and cfg.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss_sum += loss.detach() * targets.numel()
            train_elements += targets.numel()
            if accelerator.sync_gradients:
                global_step += 1
                if (cfg.wandb_project and cfg.log_every_steps > 0
                        and global_step % cfg.log_every_steps == 0):
                    batch_loss = accelerator.reduce(loss.detach(), reduction="mean")
                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "train_loss": batch_loss.item(),
                                "epoch": epoch,
                                "lr": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )
            progress.set_postfix(
                loss=f"{loss.item():.6f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_totals = torch.stack((train_loss_sum, train_elements))
        train_totals = accelerator.reduce(train_totals, reduction="sum")
        train_loss = (train_totals[0] / train_totals[1].clamp_min(1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=accelerator.device)
        val_elements = torch.zeros((), device=accelerator.device)
        # raw MSE/count, DM/TKE/MVPE sums/count, three SPS branch sums,
        # coverage count, SPS element count
        score_totals = torch.zeros(11, device=accelerator.device)
        val_progress = tqdm(
            val_loader,
            disable=not accelerator.is_local_main_process,
            desc=f"Val   {epoch:03d}",
            leave=True,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for inputs, targets in val_progress:
                targets_raw = targets
                inputs, targets = val_norm.preprocess(inputs, targets)
                predictions = model(inputs)
                batch_val_sum = F.mse_loss(predictions, targets, reduction="sum")
                val_loss_sum += batch_val_sum
                val_elements += targets.numel()

                predictions_raw = val_norm.postprocess_pred(predictions)
                batch_scores = _scoring_metric_totals(
                    predictions_raw, targets_raw, sub_s_real=cfg.sub_s)
                score_totals += batch_scores
                batch_count = batch_scores[5].clamp_min(1)
                batch_rel_l2 = (batch_scores[2] / batch_count).item()
                batch_tke = (batch_scores[3] / batch_count).item()
                val_progress.set_postfix(
                    loss=f"{(batch_val_sum / targets.numel()).item():.6f}",
                    rel_l2_score=f"{_error_score(batch_rel_l2):.2f}",
                    tke_score=f"{_error_score(batch_tke):.2f}",
                )
        val_totals = accelerator.reduce(
            torch.stack((val_loss_sum, val_elements)), reduction="sum")
        score_totals = accelerator.reduce(score_totals, reduction="sum")
        val_loss = (val_totals[0] / val_totals[1].clamp_min(1)).item()
        val_raw_mse = (score_totals[0] / score_totals[1].clamp_min(1)).item()
        val_rel_l2 = (score_totals[2] / score_totals[5].clamp_min(1)).item()
        val_tke = (score_totals[3] / score_totals[5].clamp_min(1)).item()
        val_mvpe = (score_totals[4] / score_totals[5].clamp_min(1)).item()
        sps_dm = (score_totals[6] / score_totals[10].clamp_min(1)).item()
        sps_tke = (score_totals[7] / score_totals[10].clamp_min(1)).item()
        sps_mvpe = (score_totals[8] / score_totals[10].clamp_min(1)).item()
        val_sps = 0.5 * sps_dm + 0.3 * sps_tke + 0.2 * sps_mvpe
        val_coverage = (score_totals[9] / score_totals[10].clamp_min(1)).item()
        rel_l2_score = _error_score(val_rel_l2)
        tke_score = _error_score(val_tke)
        mvpe_score = _error_score(val_mvpe)
        sps_score = 100.0 / (1.0 + math.exp(-max(-60.0, min(60.0, val_sps))))

        metrics = {
            "epoch": epoch,
            "train_loss_epoch": train_loss,
            "val_loss": val_loss,
            "val_raw_mse": val_raw_mse,
            "val_rel_l2": val_rel_l2,
            "val_tke_rel_l2": val_tke,
            "val_mvpe_rel_l2": val_mvpe,
            "val_sps": val_sps,
            "val_coverage": val_coverage,
            "val_rel_l2_score": rel_l2_score,
            "val_tke_score": tke_score,
            "val_mvpe_score": mvpe_score,
            "val_sps_score": sps_score,
            "val_quality_score": (rel_l2_score + tke_score + mvpe_score + sps_score) / 4.0,
            "lr": scheduler.get_last_lr()[0],
        }
        if cfg.wandb_project and accelerator.is_main_process:
            accelerator.log(metrics, step=global_step)

        best_marker = "  <-- best" if val_loss < best_val else ""
        accelerator.print(
            f"\nEpoch {epoch:03d}{best_marker}\n"
            f"  Losses : train={train_loss:.6f}  val={val_loss:.6f}  "
            f"raw_mse={val_raw_mse:.6f}\n"
            f"  Scores : quality={metrics['val_quality_score']:.2f}  "
            f"rel_l2={rel_l2_score:.2f}  tke={tke_score:.2f}  "
            f"mvpe={mvpe_score:.2f}  sps={sps_score:.2f}\n"
            f"  Errors : rel_l2={val_rel_l2:.4f}  tke={val_tke:.4f}  "
            f"mvpe={val_mvpe:.4f}  sps_raw={val_sps:.4f}  "
            f"coverage={val_coverage:.4f}\n"
            f"  Train  : step={global_step}  lr={metrics['lr']:.3e}"
        )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state = _checkpoint(
                cfg, accelerator, model, optimizer, scheduler, epoch, global_step,
                train_loss, val_loss, train_norm, val_norm)
            accelerator.save(state, save_dir / "last.pth")
            if val_loss < best_val:
                best_val = val_loss
                accelerator.save(state, save_dir / "best.pth")
            if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
                accelerator.save(state, save_dir / f"epoch_{epoch:03d}.pth")
        accelerator.wait_for_everyone()

    accelerator.print(f"Done. Best validation MSE: {best_val:.6f}; outputs: {save_dir}")
    if cfg.wandb_project and accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()
