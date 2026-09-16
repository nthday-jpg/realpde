# submission_v1 — Reference 1-step TTT

Promoted from `submission_template.py` (`ReferenceTTTModel` + `TinyForecaster`).

- **Adaptation:** one SGD step (`lr=1e-3`) on the cached previous input vs the
  revealed previous target, then predict current input with no grads.
  `reset_ttt_state()` restores checkpoint weights and clears the cache.
- **Base model:** `TinyForecaster` (near-identity residual conv) when no
  `model.pth` is present; loads `model.pth` (`state_dict` or
  `{"model_state_dict": ...}`) when shipped alongside.
- **Shared files (injected at pack time, not stored here):** `ttt_model.py`.

```bash
# smoke test (repo root)
python local_eval.py --submission submissions/submission_v1 --data ./example_data
# with a real baseline checkpoint staged as model.pth:
#   cp /path/to/sim_real_cno.pth submissions/submission_v1/model.pth  # gitignored
python scripts/make_submission_zip.py submission_v1
```
