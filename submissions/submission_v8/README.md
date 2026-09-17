# submission_v8 — CNO + speed-conditioned q90 band (frozen weights + intervals)

- **Adaptation:** none on weights (frozen CNO, `adapt_loss: None` always).
- **Calibration:** v4's table, but `q90` is conditioned on predicted speed —
  residuals from the last `history` windows are binned by `|prev_pred|`
  magnitude (quantile edges, `n_bins`), per-channel `q90` per bin, then
  linear-interpolated at each current pixel's `|pred|` speed. Fast pixels get
  the fast-pixel quantile; no EMA. Bounds on **every** step.
- **Base model / shared files / smoke test:** same as v4 (`model.pth` via
  `load_baseline`, `TinyForecaster` fallback, `ttt_model.py` +
  `load_baseline.py` + `realpde/rpde_baselines/` injected at pack time).

```bash
python local_eval.py --submission submissions/submission_v8 --data ./example_data
python scripts/make_submission_zip.py submission_v8 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --out dist/submission_v8_cno.zip
```
