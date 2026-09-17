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
| `submission_v4` | CNO via `load_baseline` | none + online q90 calibration | rel-L2 82.6, sps 51.8 (calibrated 2/4 steps), final 63.2 | ~30MB (with ckpt) | evaluated (smoke, see Local runs) |
| `submission_v5` | CNO via `load_baseline` | none + online relative-q90 | rel-L2 82.6, sps 50.9 (calibrated 2/4 steps), final 63.3 | ~30MB (with ckpt) | evaluated (smoke, see Local runs) |
| `submission_v6` | CNO via `load_baseline` | none + fixed `pred ± bound_frac*|pred|` every step | rel-L2 77.6, sps 50.4 (4/4 steps, default w), final 72.0 | ~30MB (with ckpt) | smoke OK, see Local runs |
| `submission_v7` | CNO via `load_baseline` | none + EMA quantile band (`ema_alpha` sweep) | rel-L2 78.7, sps 51.6 (4/4 steps, default a), final 71.3 | ~30MB (with ckpt) | smoke OK, Kaggle sweep pending |
| `submission_v8` | CNO via `load_baseline` | none + speed-conditioned q90 (interp, no EMA) | rel-L2 77.6, sps 51.5 (4/4 steps), final 70.4 | ~30MB (with ckpt) | smoke OK, Kaggle run pending (§5d) |

## Changelog

- `submission_v2_cno.zip` (v2 + `sim_real_cno.pth`, `base_model: cno`): example_data smoke rel-L2 81.7 / final 60.2 on CPU (time_score not meaningful locally; GPU eval will differ). Zip ~30MB, under cap.
- `submission_v3_cno.zip` (v3 + `sim_real_cno.pth`): example_data smoke rel-L2 82.6 / final 63.2 on CPU, 8.5s/step (10x faster than v2's 86s/step — the cost of 5 adapt steps). Zip ~30MB, under cap.
- Contract decision (2026-09-16): checked `submission_template.py` @ `959849f` (init) — the reference returns `pred_norm` on `self.device` with no input-device transfer. All shipped submissions honor that; device handling lives harness-side only (`local_eval.py --device` + stats/tensors `.to(device)`, numerically neutral, timed region untouched).
- v4 + harnessed bounds (2026-09-16): `submission_v4` returns `info["lower"/"upper"]` (normalized, per metrics.md's "if you do not return lower/upper arrays" clause); `local_eval.py` now collects, denormalizes, and scores them (missing steps fall back to the default band per-step). `submission_v4_cno.zip` smoke: rel-L2 82.6 / sps 51.8 (vs v3's 50.5 default) / final 63.2 on example_data. Next: Kaggle real-30 GPU run via notebook (`VARIANT='submission_v4'`).
- v5 relative calibration (2026-09-16): `submission_v5` scales the band by local magnitude (`rel_res = abs(t-p)/(abs(p)+1e-6)`, `width = q90*abs(pred)`). `submission_v5_cno.zip` smoke: sps 50.9 / final 63.3 on example_data — toy set too small to separate v4/v5; real-30 GPU run decides.
- v6 fixed-band probe (2026-09-17): `submission_v6` = v3 frozen CNO + fixed `pred ± bound_frac*|pred|` on every step (`bound_frac: 0.05` reproduces the scorer default; `--set bound_frac=` overrides at pack time, no code edit). Minimal test of "is the default too narrow". Smoke (TinyForecaster fallback): sps 50.36 / final 72.04, bounds on 4/4 steps. Sweep/pack via `notebook/sps_bound_kaggle.ipynb`.

## Local runs (`local_eval.py --data ./example_data`, synthetic, NOT leaderboard-comparable)

### v3, TinyForecaster fallback (no `model.pth`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 79.514 | 78.456 | 88.270 | 67.385 | 50.513 | 72.828 | 171ms |

### v3 + `sim_real_cno.pth` (unzipped `submission_v3_cno.zip`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 82.645 | 69.734 | 90.362 | 22.625 | 50.521 | 63.178 | 8.5s |

### v4, TinyForecaster fallback (no `model.pth`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 77.722 | 78.255 | 87.492 | 52.742 | 51.596 | 69.561 | 585ms |

Intervals active on 2/4 steps (first step of each trajectory has no table yet).

### v4 + `sim_real_cno.pth` (unzipped `submission_v4_cno.zip`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 82.645 | 69.734 | 90.362 | 21.251 | 51.844 | 63.167 | 10.0s |

Same frozen point predictions as v3 (accuracy identical); SPS 51.8 vs 50.5 from calibrated intervals on 2/4 steps.

### v5, TinyForecaster fallback (no `model.pth`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 77.753 | 77.714 | 87.574 | 61.970 | 50.868 | 71.176 | 275ms |

### v5 + `sim_real_cno.pth` (unzipped `submission_v5_cno.zip`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 82.645 | 69.734 | 90.362 | 22.894 | 50.862 | 63.300 | 8.3s |

### v6, TinyForecaster fallback (no `model.pth`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 77.641 | 77.796 | 87.566 | 66.829 | 50.358 | 72.038 | 180ms |

Bounds on 4/4 steps (all-or-none satisfied, `bound_frac: 0.05` = scorer default width). Width sweep on example_data (normalized-space bands, `§4 == §5` exactly): 0.05 → 50.29, 0.10 → 50.55, 0.20 → 50.98, 0.30 → 51.33, 0.50 → 51.79 — wider wins directionally; real-30 GPU run in `notebook/sps_bound_kaggle.ipynb` decides the pack width.

### v7, TinyForecaster fallback (no `model.pth`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 78.708 | 79.322 | 87.858 | 58.766 | 51.608 | 71.252 | 359ms |

Bounds on 4/4 steps (`ema_alpha: 0.30` default). Kaggle sweep over `ALPHAS = [0.10, 0.30, 0.50, 0.70, 1.00]` + v5 one-row comparison in notebook §5c; §6 packs one zip per alpha.

### v8, TinyForecaster fallback (no `model.pth`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 77.558 | 78.652 | 86.245 | 58.065 | 51.475 | 70.399 | 380ms |

Bounds on 4/4 steps (speed-binned q90, `history: 2`, `table_frames: 5`, `n_bins: 10`). Kaggle run in notebook §5d on the same real-30 split; §6 packs `dist/submission_v8_cno.zip`.

## Staged real-data runs (Kaggle GPU, `scripts/stage_real30.py` 30 traj / seed 42 — diagnostic, NOT leaderboard)

### v3 + `sim_real_cno.pth` on real-30 (2026-09-16)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 95.856 | 75.447 | 96.119 | 89.051 | 54.535 | 82.201 | 11ms |

Caveats: stats were fit on the same 30 trajectories (self-normalized, not official `mean_std_real.pt`); trajectories truncated to 200 frames. Still, the pattern matches Codabench: accuracy excellent, SPS (54.5, default ±5% band) the clear laggard → motivates v4 calibration.

### v6 width sweep on real-30 (2026-09-17, Kaggle GPU, `notebook/sps_bound_kaggle.ipynb`)

Same frozen CNO, fixed normalized-space bands `predn ± w*|predn|`, official SPS (§4 == §5 exactly):

| half-width | total-width | sps_score | coverage | mean_nil |
|---|---|---|---|---|
| 0.050 | 0.100 | 54.33 | 0.2821 | 0.090 |
| 0.100 | 0.200 | 55.68 | 0.4202 | 0.181 |
| 0.200 | 0.400 | 56.50 | 0.5679 | 0.361 |
| 0.300 | 0.600 | 56.72 | 0.6548 | 0.542 |
| 0.500 | 1.000 | 56.81 | 0.7590 | 0.903 |

Monotone gains with fast decay (+1.35 / +0.82 / +0.22 / +0.09) — default ±5% far too narrow, peak at or beyond w=0.5. Next: extend sweep to 0.75–1.50 before packing; ~24% uncovered at w=0.5 hints at a stagnation-region floor (`w*|pred| + f`). Same staged-data caveats as the v3 real-30 run above.

### v4 adaptive on real-30 (2026-09-18, Kaggle GPU, notebook §5b — 270 steps)

| sps_score | coverage | mean_nil | steps-with-bounds |
|---|---|---|---|
| 59.97 | 0.7948 | 0.381 | 270/270 |

Configs: `sim_real_cno.pth`, `coverage: 0.90`, `history: 2` windows, `table_frames: 5`, `fallback_frac: 0.05`. Same 30-traj staged split as the v6 sweep above. Adaptive beats the best fixed width (56.81 @ w=0.5) on score AND sharpness (0.381 vs 0.903 mean_nil at higher coverage 0.79 vs 0.76) — per-channel q90 over recent residuals adapts the width to the flow instead of paying a uniform fraction. Next: v7 (EMA version of the same band) vs v5 (relative-q90) on this split.

## Leaderboard (Codabench, real `test_real`)
### `submission_v3_cno.zip` — v3 CNO, no adaptation (2026-09-16)

| rel_l2 | tke | mvpe | time | sps | final |
|---|---|---|---|---|---|
| 94.514 | 73.804 | 93.099 | 86.658 | 15.869 | **72.837** |

Frozen fine-tuned CNO is strong on point/probe accuracy (rel-L2 94.5, MVPE 93.1) and fast (time 86.7), but SPS 15.9 drags the mean — the default ±5%-magnitude interval is far too narrow for frozen point predictions. Takeaway: keep the CNO base, add calibrated uncertainty intervals (SPS fix) before spending budget on weight adaptation.
