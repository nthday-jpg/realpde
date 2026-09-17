# submission_v7 — CNO + EMA quantile calibration (frozen weights + intervals)

- **Adaptation:** none on weights (frozen CNO, `adapt_loss: None` always).
- **Calibration:** v4's absolute band with infinite memory — per-step
  `q90 = quantile(|prev_target − prev_pred|, 0.90)` folded into an EMA
  (`ema = alpha*q + (1−alpha)*ema`), `lower/upper = pred ∓/± ema` in
  normalized space, returned on **every** step. `alpha = 1.0` is memoryless.
- **Base model / shared files / smoke test:** same as v4/v6 (`model.pth` via
  `load_baseline`, `TinyForecaster` fallback, `ttt_model.py` +
  `load_baseline.py` + `realpde/rpde_baselines/` injected at pack time).

```bash
python local_eval.py --submission submissions/submission_v7 --data ./example_data
python scripts/make_submission_zip.py submission_v7 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --set ema_alpha=0.30 --out dist/submission_v7_a030.zip
```
