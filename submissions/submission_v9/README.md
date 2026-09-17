# submission_v9 — CNO + 1-step SGD (plain TTT ablation, no intervals)

- **Adaptation:** one SGD step per `ttt_step` on the revealed previous
  `(input, target)` pair (reference loop from `submission_template.py`),
  then predict. `adapt_loss` is the step's MSE.
- **Intervals:** none returned — scorer grades the default `±5%` band.
  Compare v3 (frozen) → v9 (adapted) → v10 (adapted + calibrated).
- **Base model / shared files:** `model.pth` via `load_baseline`
  (`policy.yaml: base_model`, default `cno`); `TinyForecaster` fallback.

```bash
python local_eval.py --submission submissions/submission_v9 --data ./example_data
python scripts/make_submission_zip.py submission_v9 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --out dist/submission_v9_cno.zip
```
