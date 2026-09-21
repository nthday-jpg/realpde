# Notebook guide — read this instead of opening .ipynb files

All training/eval notebooks run on **Kaggle GPU** (laptop has no CUDA).
Local work = write/edit code; Kaggle work = run cells. Never `pip install`
heavy deps locally.

## Standard cell pattern (every Kaggle training/eval notebook follows this)

1. **Clone/pull repo** → `REPO_DIR=/kaggle/working/realpde` (public, no token).
2. **`%cd` repo**, then `uv python pin 3.10` + `uv sync --python 3.10`.
3. **Install into kernel**: `{sys.executable} -m pip install -q -e . --no-deps
   --ignore-requires-python`, then prepend `<repo>/src` to `sys.path`
   (kernel is 3.12, package is pure-Python — plain install refuses without the flag).
4. **GPU check** (`torch.cuda.is_available()`).
5. **CONFIG cell (env vars)** — the only cell you edit; everything below reads it.
6. **`DATA_ROOT` = `/kaggle/input/datasets/nthday/realpde`** (fallbacks `./data`,
   `data` for local runs), with `{baseline,test,train_real,train_sim}/` directly
   underneath. `resolve_h5_dir()` handles flat (`train_sim/*.h5`) and nested
   (`train_sim/train_sim/*.h5`) layouts; checkpoints via `(root/'baseline').rglob()
7. **Preload in processes**: `scripts/cache_dataset.py` automatically sets
   `PDE_PRELOAD_BACKEND=process`; tune process count with `CACHE_WORKERS`
   (h5py serializes threads, which otherwise causes roughly 10% CPU usage).
8. **Work cells → save to `/kaggle/working/`** (only dir that persists/downloads).

## Notebooks

| Notebook | Purpose | Key CONFIG | Output |
|---|---|---|---|
| `baseline_kaggle.ipynb` | Score all 8 shipped baselines, 4 settings (A sim→sim, B sim→real zero-shot, C finetuned→real, D finetuned→sim). Real eval = `train_real/` only, 40% of files (seeded-shuffled, staged symlinks in `/tmp/realpde_sub/`, cached). Direct teacher-forcing, raw-space MSE + rel-L2. | `BASELINE_BATCH_SIZE=4`, `BASELINE_SAMPLE_FRAC=0.4`, `PREFER_FP16_FNO=1`, `FILE_SEED=42` | `/kaggle/working/baseline_matrix.csv` |
| `pretrain_kaggle.ipynb` | Train `unet` from scratch via `scripts/trainer.py` + `accelerate launch` (multi-GPU OK), then `scripts/eval_pretrain.py` + `local_eval.py` smoke tests. Normalizes with per-split stats (no leakage); ckpts carry `norm_train`/`norm_val` + `mean_std_{train,val}.pt`. | `DATA_PATH`, `MODEL_NAME=unet`, `LR=1e-3`, `EPOCHS=50`, `VAL_FRAC=0.1`, `SEED=42`, `SAVE_DIR=/kaggle/working/checkpoints` | `checkpoints/{best,final,epoch_NNN}.pth` |
| `train_fno_kaggle.ipynb` | Train a configurable `FNO3d` from scratch or resume a compatible checkpoint via `scripts/train_fno.py` + HF Accelerate. Architecture, optimizer, windows, checkpointing, and optional W&B settings are all in one CONFIG cell. | `FNO_MODES{1,2,3}=4,12,16`, `FNO_N_LAYERS=4`, `FNO_WIDTH=64`, `FNO_PADDING=6`, `LR=1e-4`, `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=4` | `fno_checkpoints/{best,last,epoch_NNN}.pth` + normalization stats |
| `continue_cno_kaggle.ipynb` | **CNO-sim undertrained check**: thin launcher for `scripts/finetune_baseline.py` via `accelerate launch`. Run A resumes shipped `sim_cno.pth` **on `train_sim`** (same distribution — falling val = undertrained); optional Run B finetunes same ckpt on `train_real/`. Verdict cell scores shipped vs continued with the baseline scorer. | `SIM_CNO` (auto), `LR=1e-4`, `EPOCHS=20`, `BATCH_SIZE=2` (1 if OOM), `SAVE_DIR=/kaggle/working/cno_sim_resume` (+ `cno_real_ft`), `REAL_DATA_PATH` (+ `REAL_FRAC=0.2`) for the built-in post-train test (continued-best vs before-train on real) | `cno_sim_resume/{best,final,last}.pth` |
| `inspect_*.ipynb`, `visualize.ipynb` | Dataset inspection/plots, CPU-OK. | — | figures only |
| `eval_kaggle.ipynb` | Editor-first eval runner: smoke-test every variant via `local_eval.py`, stage real-30 + score CNO variants on GPU, SPS symmetry probe (v4 symmetric q90 vs asymmetric [q05,q95] in one run via `local_eval.sps_component_breakdown`), pack Codabench zips. Replaces `sps_bound_kaggle.ipynb` (removed 2026-09-19; its sweep logic now lives in `local_eval`). | `VARIANT='submission_v4'` (real-30 cell), `N_FILES=30`, `N_FRAMES=200`, `SEED=42` | `local_eval` logs + `/kaggle/working/real30/`, `dist/*.zip` |

## Rules for new training/eval notebooks

- Training logic lives in scripts (`scripts/trainer.py` for unet, `scripts/train_fno.py` for configurable
  FNO training, and `scripts/finetune_baseline.py` for shipped CNO/FNO/Transolver via `load_baseline` + accelerate); notebooks only set
  env config and launch `accelerate`. Copy `continue_cno_kaggle.ipynb` (finetune pattern) or `pretrain_kaggle.ipynb`
  (`accelerate` pattern); keep cells 1–6 identical so datasets/mounts keep working.
- All training/eval runs in normalized space (`src/realpde/datasets/normalizer.py:PDENormalizer`,
  same convention as `local_eval.py`): train stats fit on train files only, val/test stats on
  that split only — never pool splits. Loss is normalized-space MSE; leaderboard-style metrics
  are always denormalized to raw space first (`postprocess_pred`).
- Config **only** via `os.environ.setdefault(...)` in one CONFIG cell.
- Save checkpoints as `{'model_state_dict': ..., 'epoch': ..., 'val_loss': ...}`
  (+ `optimizer_state_dict` for chained runs); keep `*cno*.pth`-style arch names
  so `detect_model_type()` works.
- Chaining runs: set `RESUME_CKPT=<prior best.pth>` — loop auto-restores
  optimizer/epoch when `optimizer_state_dict` is present.
- Dataset cache: `scripts/cache_dataset.py` dumps all windows to
  `/kaggle/working/cache_<split>.pt` once (~5GB fp32 for train_sim, persists
  across sessions); `scripts/finetune_baseline.py` uses it via `DATA_CACHE` and
  skips the slow per-run `.h5` reads. Re-cache only if window knobs change.
- Single GPU default; only `baseline_kaggle.ipynb` has the DDP scorer
  (subprocess workers, one/GPU). Don't copy its DDP block into training loops.
- After adding a notebook, add one row to the table above — that's the contract
  that lets the next session skip reading the .ipynb.
