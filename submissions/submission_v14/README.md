# submission_v14 — full-gradient TTT (MSE + temporal-difference) + TKE maps

- **Adaptation:** every `ttt_step` with a revealed previous pair takes
  `ttt_steps` (default 3) gradient steps on **all** weights (full update, not
  LoRA, not 1-step) with
  `loss = MSE(pred_prev, prev_target) + td_lambda * MSE(Δpred, Δtgt)`,
  where `Δ` is the 1-step temporal difference on (u, v). MSE anchors the
  level, TD anchors the dynamics so adaptation cannot trade temporal
  variability for instantaneous error — the failure mode the TKE scalar hides.
- **Calibration:** v4/v10 online per-channel `q90` band on every step
  (`return_bounds: true`; set false for a point-only ablation).
- **TKE spatial logging:** `tke_maps.py` replays the streaming loop and logs
  the spatial distribution `KE(x) = 1/2[Var_t(u) + Var_t(v)]` per window, with
  the revealing panel `KE_TTA − KE_target` (not just the scalar TKE error).
- **Base model / shared files:** `model.pth` via `load_baseline`
  (`policy.yaml: base_model`, default `cno`); `TinyForecaster` fallback.

```bash
python local_eval.py --submission submissions/submission_v14 --data ./example_data
python submissions/submission_v14/tke_maps.py --submission submissions/submission_v14 \
  --data ./example_data --out tke_out_v14 --max-windows 4
python scripts/make_submission_zip.py submission_v14 \
  --with-model data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth \
  --out dist/submission_v14_cno.zip
```
