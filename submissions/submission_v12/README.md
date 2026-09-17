# submission_v12 — FNO + online q90 calibration (frozen weights + intervals)

Same band as `submission_v4`, different backbone: frozen FNO point predictions
plus the per-step `q90(|resid|)` interval table over the revealed previous
pair (`history: 2`, `table_frames: 5`, `coverage: 0.90`, `fallback_frac: 0.05` —
identical knobs, so v4-vs-v12 isolates the CNO→FNO backbone effect on both
point accuracy and SPS).

- **Adaptation:** none on weights (frozen FNO, `adapt_loss: None` always).
- **Calibration:** v4's table, unchanged (`submission.py` is v4's logic retagged).
- **Base model:** `sim_real_fno` checkpoint via `load_baseline`
  (`policy.yaml: base_model: fno`); `TinyForecaster` fallback.
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
