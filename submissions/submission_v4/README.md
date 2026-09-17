# submission_v4 — CNO + online q90 calibration (frozen weights + intervals)

- **Adaptation:** none on weights (frozen CNO, `adapt_loss: None` always).
- **Calibration:** per-step table lookup on the revealed previous pair —
  `residual = abs(prev_target - prev_pred)`, per-channel `q90 = quantile(residual, 0.90)`
  over the last `history` windows, `lower/upper = pred ∓/± q90` for the current
  prediction. Bounds ride in `info["lower"/"upper"]` (normalized space); first
  step omits them (scorer default applies).
- **Base model:** baseline checkpoint `model.pth` via `load_baseline`
  (`policy.yaml: base_model`, default `cno`); `TinyForecaster` fallback.
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
