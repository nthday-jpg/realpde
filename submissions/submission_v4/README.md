# submission_v4 — CNO + online q90 calibration (frozen weights + intervals)

- **Adaptation:** none on weights (frozen CNO, `adapt_loss: None` always).
- **Calibration:** per-step table lookup on the revealed previous pair —
  `residual = abs(prev_target - prev_pred)`, per-channel `q90 = quantile(residual, 0.90)`
  over the last `history` windows, `lower/upper = pred ∓/± q90` for the current
  prediction. Bounds ride in `info["lower"/"upper"]` (normalized space); first
  step omits them (scorer default applies).
- **Base model:** baseline checkpoint `model.pth` via `load_baseline`
  (`policy.yaml: base_model`, default `cno`); `TinyForecaster` fallback.
- **Symmetry probe (2026-09-19, real-30 Kaggle):** asymmetric `[q05, q95]` band
  scores sps 59.91 / coverage 0.8196 / mean_nil 0.424 vs symmetric q90's
  59.92 / 0.8211 / 0.425 — no difference (u `q95/|q05| = 1.130`, v `= 0.984`).
  Keep the symmetric band.
- **Shared files (injected at pack time, not stored here):** `ttt_model.py`,
  `load_baseline.py`, `realpde/rpde_baselines/`.

```bash
# smoke test without weights (TinyForecaster fallback)
python local_eval.py --submission submissions/submission_v4 --data ./example_data
# pack with real weights
python scripts/make_submission_zip.py submission_v4 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --out dist/submission_v4_cno.zip
```
