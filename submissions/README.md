# Submissions — research index

Editor-first flow: implement each variant as real files under
`submissions/<variant>/`, `git push`, then on Kaggle `git pull` and run
`local_eval.py --submission submissions/<variant>`. Never hand-write
`submission.py` inside notebook cells.

Each `submission_vN/` stays lean (`submission.py` + optional `policy.yaml` +
local `README.md`); shared files (`ttt_model.py`, `load_baseline.py`,
`realpde/rpde_baselines/`) are injected by `scripts/make_submission_zip.py`
at pack time so the Codabench zip is self-contained with `submission.py` at
its root. `model.pth` is never committed (see `.gitignore`).

| Variant | Base model | Adapt method | local_eval (example_data) | Zip size | Status |
|---|---|---|---|---|---|
| `submission_v1` | TinyForecaster / `model.pth` | ref 1-step SGD (`ReferenceTTTModel`) | rel-L2 78.6, 389ms/step, final 71.0 | 5KB (no ckpt) | evaluated (smoke) |
| `submission_v2` | TinyForecaster / baseline via `load_baseline` | bounded controller (`rule` default) | rel-L2 71.9, 181ms/step, final 64.9 | ~0.6MB (no ckpt) | evaluated (smoke) |
| `submission_v3` | CNO via `load_baseline` | none (predict-only) | rel-L2 82.6, 8.5s/step CPU, final 63.2 | ~30MB (with ckpt) | ✅ submitted — Codabench final 72.84 (see Leaderboard below) |

## Changelog

- `submission_v2_cno.zip` (v2 + `sim_real_cno.pth`, `base_model: cno`): example_data smoke rel-L2 81.7 / final 60.2 on CPU (time_score not meaningful locally; GPU eval will differ). Zip ~30MB, under cap.
- `submission_v3_cno.zip` (v3 + `sim_real_cno.pth`): example_data smoke rel-L2 82.6 / final 63.2 on CPU, 8.5s/step (10x faster than v2's 86s/step — the cost of 5 adapt steps). Zip ~30MB, under cap.
- Contract decision (2026-09-16): checked `submission_template.py` @ `959849f` (init) — the reference returns `pred_norm` on `self.device` with no input-device transfer, and `local_eval.py` hardcodes CPU. Reverted all deviations: kit files (`local_eval.py`, `submission.py`, `submission_template.py`, `agentic_demo/`) are pristine, and v1/v2/v3 return exactly per template. Consequence: `local_eval` runs CPU-only everywhere, so the Kaggle real-30 CNO run takes ~40+ min (reduce `--frames` for iteration).

## Local runs (`local_eval.py --data ./example_data`, synthetic, NOT leaderboard-comparable)

### v3, TinyForecaster fallback (no `model.pth`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|
| 79.514 | 78.456 | 88.270 | 67.385 | 50.513 | 72.828 | 171ms |

### v3 + `sim_real_cno.pth` (unzipped `submission_v3_cno.zip`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|
| 82.645 | 69.734 | 90.362 | 22.625 | 50.521 | 63.178 | 8.5s |

## Leaderboard (Codabench, real `test_real`)
### `submission_v3_cno.zip` — v3 CNO, no adaptation (2026-09-16)

| rel_l2 | tke | mvpe | time | sps | final |
|---|---|---|---|---|---|
| 94.514 | 73.804 | 93.099 | 86.658 | 15.869 | **72.837** |

Frozen fine-tuned CNO is strong on point/probe accuracy (rel-L2 94.5, MVPE 93.1) and fast (time 86.7), but SPS 15.9 drags the mean — the default ±5%-magnitude interval is far too narrow for frozen point predictions. Takeaway: keep the CNO base, add calibrated uncertainty intervals (SPS fix) before spending budget on weight adaptation.
