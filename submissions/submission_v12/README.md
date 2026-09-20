# submission_v12 — FNO + frozen q90 band (compute once, reuse everywhere)

Frozen FNO point predictions plus a q90 band computed once and reused: the
first `warmup_windows` revealed pairs fill the table like v4 (warmup steps are
online-quality), then the per-channel q90 freezes run-wide — no further
table/quantile work. Promoted from `submission_v13` (removed) after the frozen
band beat the online band on real-30 (sps 61.05 vs 60.53, final 83.16 vs
82.80): sharpness won over coverage.

- **Adaptation:** none on weights (frozen FNO, `adapt_loss: None` always).
- **Calibration:** v4 table during warmup, frozen `pred ± q90` after
  (`warmup_windows: 5`, `coverage: 0.90`, `history: 2`, `table_frames: 5`);
  `TinyForecaster` fallback, bounds every step.
- **Base model:** `sim_real_fno` checkpoint via `load_baseline`
  (`policy.yaml: base_model: fno`).
- **Size:** the fp32 FNO checkpoint (~403 MB) exceeds the 256 MB cap — pack
  with the fp16 checkpoint (`sim_real_fno_fp16.pth`, ~201 MB), which
  `load_baseline` unpacks transparently.
- **Shared files (injected at pack time, not stored here):** `ttt_model.py`,
  `load_baseline.py`, `realpde/rpde_baselines/`.

```bash
# smoke test without weights (TinyForecaster fallback)
python local_eval.py --submission submissions/submission_v12 --data ./example_data
# smoke test with real weights: stage the checkpoint as model.pth first
cp data/baseline_checkpoints/sim_real_ft/sim_real_fno_fp16.pth submissions/submission_v12/model.pth
python local_eval.py --submission submissions/submission_v12 --data ./example_data
# pack with real weights (fp16 to stay under the cap)
python scripts/make_submission_zip.py submission_v12 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_fno_fp16.pth \
  --out dist/submission_v12_fno.zip
```
