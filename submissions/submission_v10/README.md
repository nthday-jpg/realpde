# submission_v10 — CNO + 1-step SGD + v4 calibration band

- **Adaptation:** one SGD step per `ttt_step` on the revealed previous pair
  (same loop as v9), then predict.
- **Calibration:** v4's online per-channel `q90` band on the current
  prediction, returned on **every** step (`adapt_loss` still reported).
- **Ladder:** v3 (frozen) → v9 (adapted) → v10 (adapted + calibrated)
  isolates each ingredient; v4 vs v10 isolates adaptation under calibration.

```bash
python local_eval.py --submission submissions/submission_v10 --data ./example_data
python scripts/make_submission_zip.py submission_v10 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --out dist/submission_v10_cno.zip
```
