# submission_v13 — FNO + frozen q90 band (compute once, reuse everywhere)

v12's band minus the per-step quantile: the first `warmup_windows` revealed
pairs fill the table exactly like v4/v12 (warmup steps are v12-quality), then
the per-channel q90 freezes and is reused for every later step of every
trajectory. Tests whether the stationary q90 can be amortized to ~zero
calibration cost on the stronger FNO backbone — v13 vs v12 isolates
frozen-vs-online with the backbone held fixed.

- **Adaptation:** none on weights (frozen FNO, `adapt_loss: None` always).
- **Calibration:** v4 table during warmup, frozen `pred ± q90` after
  (`warmup_windows: 5`); `TinyForecaster` fallback, bounds every step.
- **Base model:** `sim_real_fno` checkpoint via `load_baseline`
  (`policy.yaml: base_model: fno`).
- **Size:** the fp32 FNO checkpoint (~403 MB) exceeds the 256 MB cap — pack
  with the fp16 checkpoint (`sim_real_fno_fp16.pth`, ~201 MB), which
  `load_baseline` unpacks transparently.
- **Shared files (injected at pack time, not stored here):** `ttt_model.py`,
  `load_baseline.py`, `realpde/rpde_baselines/`.

```bash
# smoke test without weights (TinyForecaster fallback)
python local_eval.py --submission submissions/submission_v13 --data ./example_data
# smoke test with real weights: stage the checkpoint as model.pth first
cp data/baseline_checkpoints/sim_real_ft/sim_real_fno_fp16.pth submissions/submission_v13/model.pth
python local_eval.py --submission submissions/submission_v13 --data ./example_data
# pack with real weights (fp16 to stay under the cap)
python scripts/make_submission_zip.py submission_v13 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_fno_fp16.pth \
  --out dist/submission_v13_fno.zip
```
