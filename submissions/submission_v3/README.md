# submission_v3 — CNO, no adaptation (predict-only ablation)

- **Adaptation:** none. `ttt_step` predicts under `no_grad` and returns
  `adapt_loss: None`. Answers "how much does TTT add over the frozen
  fine-tuned baseline?" when compared against v1/v2 with the same weights.
- **Base model:** baseline checkpoint `model.pth` via `load_baseline`
  (hint from `policy.yaml: base_model`, default `cno`); `TinyForecaster`
  fallback when no checkpoint is staged.
- **Shared files (injected at pack time, not stored here):** `ttt_model.py`,
  `load_baseline.py`, `realpde/rpde_baselines/`.

```bash
# smoke test without weights (TinyForecaster fallback)
python local_eval.py --submission submissions/submission_v3 --data ./example_data
# pack with real weights (policy already says base_model: cno)
python scripts/make_submission_zip.py submission_v3 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --out dist/submission_v3_cno.zip
```
