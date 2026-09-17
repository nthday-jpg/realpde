# submission_v6 — v3 frozen CNO + fixed relative band (minimal SPS probe)

- **Adaptation:** none (same predict-only path as v3, `adapt_loss: None`).
- **Intervals:** fixed `lower/upper = pred ∓/± bound_frac*|pred|` in normalized
  space, returned on **every** step (ingestion enforces all-or-none).
  `bound_frac: 0.05` reproduces the scorer default; widen to test coverage.
- **Base model / shared files:** same as v3 (`model.pth` via `load_baseline`,
  `TinyForecaster` fallback, `ttt_model.py` + `load_baseline.py` +
  `realpde/rpde_baselines/` injected at pack time).

```bash
# smoke test without weights (TinyForecaster fallback)
python local_eval.py --submission submissions/submission_v6 --data ./example_data
# sweep the width without editing code (--set rewrites policy.yaml in the zip)
python scripts/make_submission_zip.py submission_v6 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --set bound_frac=0.20 --out dist/submission_v6_w20.zip
```
