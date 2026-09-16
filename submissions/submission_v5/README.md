# submission_v5 — CNO + online relative-q90 calibration

- **Adaptation:** none on weights (frozen CNO, `adapt_loss: None` always).
- **Calibration:** magnitude-scaled band — `rel_res = |prev_target - prev_pred| / (|prev_pred| + 1e-6)`,
  per-channel `q90 = quantile(rel_res, 0.90)` over the last `history` windows,
  `width = q90 * |pred|`, `lower/upper = pred ∓/± width`. Wide where flow is
  fast, tight near stagnation (v4's absolute band can't do this).
- **Base model / shared files / smoke test:** same as v4 (`model.pth` via
  `load_baseline`, `TinyForecaster` fallback, `ttt_model.py` +
  `load_baseline.py` + `realpde/rpde_baselines/` injected at pack time).

```bash
python local_eval.py --submission submissions/submission_v5 --data ./example_data
python scripts/make_submission_zip.py submission_v5 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --out dist/submission_v5_cno.zip
```
