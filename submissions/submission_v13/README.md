# submission_v13 — CNO + frozen q90 band (compute once, reuse everywhere)

v4's band minus the per-step quantile: the first `warmup_windows` revealed
pairs fill the table exactly like v4 (warmup steps are v4-quality), then the
per-channel q90 freezes and is reused for every later step of every
trajectory. Tests whether the stationary q90 can be amortized to ~zero
calibration cost.

- **Adaptation:** none on weights (frozen CNO, `adapt_loss: None` always).
- **Calibration:** v4 table during warmup, frozen `pred ± q90` after
  (`warmup_windows: 5`); `TinyForecaster` fallback, bounds every step.
- **Base model:** baseline checkpoint `model.pth` via `load_baseline`
  (`policy.yaml: base_model`, default `cno`).
- **Shared files (injected at pack time, not stored here):** `ttt_model.py`,
  `load_baseline.py`, `realpde/rpde_baselines/`.

```bash
# smoke test without weights (TinyForecaster fallback)
python local_eval.py --submission submissions/submission_v13 --data ./example_data
# smoke test with real weights: stage the checkpoint as model.pth first
cp data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth submissions/submission_v13/model.pth
python local_eval.py --submission submissions/submission_v13 --data ./example_data
# pack with real weights
python scripts/make_submission_zip.py submission_v13 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --out dist/submission_v13_cno.zip
```
