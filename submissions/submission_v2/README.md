# submission_v2 — Agentic bounded-controller TTT

Promoted from `agentic_demo/submission.py` (`AgenticTTTModel` + `policy.yaml`).

- **Adaptation:** per-step bounded action in
  `{skip_update, recalibrate, update_adapter}` driven by prev-pair rel-L2,
  error EMA slope, step index, and time budget. `update_adapter` runs
  `adapt_steps` SGD steps on the revealed previous pair (scope `bn` = norm
  layers only, or `all`); `recalibrate` maintains an additive output bias;
  `skip_update` does nothing. `mode: rule` (default) is deterministic;
  `mode: llm` uses the organizer gateway with rule fallback; `mode: fixed`
  pins `fixed_action` (e.g. `skip_update` = no-adapt ablation).
- **Base model:** `TinyForecaster` fallback; loads `model.pth` as a baseline
  checkpoint via `load_baseline` (`base_model: cno|fno|transolver` in
  `policy.yaml` when the filename carries no architecture hint).
- **Knobs:** edit `policy.yaml` offline, ship the tuned file.
- **Shared files (injected at pack time, not stored here):** `ttt_model.py`,
  `load_baseline.py`, `realpde/rpde_baselines/` (only when the submission
  imports `load_baseline`).

```bash
# smoke test (repo root, rule mode, CPU)
python local_eval.py --submission submissions/submission_v2 --data ./example_data
python scripts/make_submission_zip.py submission_v2
```
