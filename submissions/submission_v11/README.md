# submission_v11 — CNO + test-time LoRA (adapters only, no intervals)

- **Adaptation:** one SGD step per `ttt_step` on the revealed previous pair,
  restricted to low-rank Conv3d adapters (`y = conv(x) + (α/r)·B(A(x))`,
  `B` zero-init = checkpoint-identical start). Base weights frozen;
  `reset_ttt_state` restores adapters to zero. Inline torch (no `peft`).
- **Intervals:** none — scorer default band. v9 → v11 isolates full vs LoRA
  updates; CNO-only (falls back to v9-equivalent full adaptation when the
  base has no Conv3d, e.g. the `TinyForecaster` smoke path).
- **Base model / shared files:** `model.pth` via `load_baseline`
  (`policy.yaml: base_model`, default `cno`).

```bash
python local_eval.py --submission submissions/submission_v11 --data ./example_data
python scripts/make_submission_zip.py submission_v11 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --set lora_rank=8 --out dist/submission_v11_r8.zip
```
