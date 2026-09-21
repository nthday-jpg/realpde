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
| `submission_v6` | CNO via `load_baseline` | none + fixed `pred ± bound_frac*|pred|` every step | smoke OK (bounds 4/4) | ~30MB (with ckpt) | real-30: best w=0.5 sps 56.81 (see Staged) |
| `submission_v7` | CNO via `load_baseline` | none + EMA quantile band (`ema_alpha` sweep) | smoke OK (bounds 4/4) | ~30MB (with ckpt) | real-30: alpha-flat 59.96–59.98, best 1.0 (see Staged) |
| `submission_v8` | CNO via `load_baseline` | none + speed-conditioned q90 (interp, no EMA) | smoke OK (bounds 4/4) | ~30MB (with ckpt) | real-30: sps 58.72, trails v4 (see Staged) |
| `submission_v9` | CNO via `load_baseline` | 1-step SGD, no bounds (default band) | smoke OK (default band) | ~30MB (with ckpt) | real-30 full row: final 74.17 (see Staged) |
| `submission_v10` | CNO via `load_baseline` | 1-step SGD + v4 q90 band every step | smoke OK (bounds 4/4) | ~30MB (with ckpt) | real-30 full row: final 74.71 (see Staged) |
| `submission_v11` | CNO via `load_baseline` | 1-step SGD on LoRA adapters only (rank 4) | smoke OK (36 Conv3d, 329k LoRA params) | ~30MB (with ckpt) | pending Kaggle run |
| `submission_v12` | FNO via `load_baseline` | none + frozen q90 band (online warmup, then reuse) | rel-L2 82.4, sps 51.5 (warmup path, 4/4 steps), final 61.8 | ~201MB (fp16 ckpt) | evaluated (smoke); real-30 final 83.16, sps 61.05 (see Staged) |
| `submission_v14` | CNO via `load_baseline` | full update (3×SGD, all weights, MSE + TD reg), no bounds (default band) | smoke OK (default band) | ~30MB (with ckpt) | pending Kaggle run |

## Changelog

- `submission_v2_cno.zip` (v2 + `sim_real_cno.pth`, `base_model: cno`): example_data smoke rel-L2 81.7 / final 60.2 on CPU (time_score not meaningful locally; GPU eval will differ). Zip ~30MB, under cap.
- `submission_v3_cno.zip` (v3 + `sim_real_cno.pth`): example_data smoke rel-L2 82.6 / final 63.2 on CPU, 8.5s/step (10x faster than v2's 86s/step — the cost of 5 adapt steps). Zip ~30MB, under cap.
- Contract decision (2026-09-16): checked `submission_template.py` @ `959849f` (init) — the reference returns `pred_norm` on `self.device` with no input-device transfer. All shipped submissions honor that; device handling lives harness-side only (`local_eval.py --device` + stats/tensors `.to(device)`, numerically neutral, timed region untouched).
- v4 + harnessed bounds (2026-09-16): `submission_v4` returns `info["lower"/"upper"]` (normalized, per metrics.md's "if you do not return lower/upper arrays" clause); `local_eval.py` now collects, denormalizes, and scores them (missing steps fall back to the default band per-step). `submission_v4_cno.zip` smoke: rel-L2 82.6 / sps 51.8 (vs v3's 50.5 default) / final 63.2 on example_data. Next: Kaggle real-30 GPU run via notebook (`VARIANT='submission_v4'`).
- v5 relative calibration (2026-09-16): `submission_v5` scales the band by local magnitude (`rel_res = abs(t-p)/(abs(p)+1e-6)`, `width = q90*abs(pred)`). `submission_v5_cno.zip` smoke: sps 50.9 / final 63.3 on example_data — toy set too small to separate v4/v5; real-30 GPU run decides.
- v6 fixed-band probe (2026-09-17): `submission_v6` = v3 frozen CNO + fixed `pred ± bound_frac*|pred|` on every step (`bound_frac: 0.05` reproduces the scorer default; `--set bound_frac=` overrides at pack time, no code edit). Minimal test of "is the default too narrow". Sweep/pack via `notebook/sps_bound_kaggle.ipynb` (removed 2026-09-19; sweeps now via `local_eval.py --json-out`, symmetry probe in `notebook/eval_kaggle.ipynb`).
- v7 EMA band (2026-09-18): `submission_v7` = v4's absolute band with infinite memory (`ema = alpha*q + (1-alpha)*ema`, `alpha=1` memoryless). Calibration-only bench (fake tensors, no forward): v7 18.1 vs v4 52.1 ms/step — ~3× cheaper (quantile dominates; v4 sorts 2 windows, v7 one).
- v8 speed-conditioned band (2026-09-18): `submission_v8` bins residuals by `|prev_pred|` magnitude (quantile edges, `n_bins: 10`), per-channel q90 per bin, linear-interp at each current pixel's speed. No EMA. Direct test of the `C(s)` miscalibration diagnosis.
- v9/v10 adaptation ladder (2026-09-18): `submission_v9` = reference 1-step SGD on all weights (no bounds); `submission_v10` = same + v4 band. v3 → v9 → v10 isolates adaptation, then calibration.
- v13 frozen-q90 band (2026-09-19, merged into v12 same day): `submission_v13` was v12's FNO + v4-style table for the first `warmup_windows: 5` pairs, then run-wide freeze. Real-30: sps 61.05 vs online 60.53 — frozen won, so v12 now runs the frozen implementation and `submission_v13/` was removed (history kept here).
- v12 FNO+q90 (2026-09-19): `submission_v12` = frozen FNO band (`base_model: fno`, online warmup then run-wide freeze; previously v4's per-step band, superseded). fp32 ckpt (~403 MB) exceeds the cap — smoke/pack with `sim_real_fno_fp16.pth`. Run via `notebook/eval_kaggle.ipynb` (`VARIANT='submission_v12'`, `MODEL_HINT='fno'`).
- v11 test-time LoRA (2026-09-18): `submission_v11` wraps all 36 CNO Conv3ds with rank-4 adapters (`y = conv(x) + (α/r)·B(A(x))`, B zero-init), 329,616 trainable params (~4%), 1-step SGD on adapters only. Pure torch (no `peft`). Local CNO smoke: wraps/loads/adapts end-to-end. Notebook §5f pending.
- v14 full MSE+TD update + TKE maps (2026-09-21): `submission_v14` takes `ttt_steps` (default 3) full-parameter gradient steps every `ttt_step` on `MSE + td_lambda*MSE(Δpred, Δtgt)` (Δ = 1-step temporal diff on u, v), no intervals (scorer default band, v9-style isolation). Companion `tke_maps.py` replays the stream and logs per-window `KE(x) = 1/2[Var_t(u)+Var_t(v)]` maps plus the `KE_TTA − KE_target` diff panels (`ke_win*.png`, `ke_mean_diff.png`, `ke_maps.npz`); scalar cross-check matches `scoring.py` exactly. Smoke (TinyForecaster fallback): bounds 4/4, final 68.8. Next: Kaggle real-30 GPU run.

## Local runs (`local_eval.py --data ./example_data`, synthetic, NOT leaderboard-comparable)

TinyForecaster-fallback (no-ckpt) score rows removed 2026-09-18 — superseded by staged real-30 runs below; fallback smoke is pass/fail only. Only CNO-weight rows are kept.

### v3 + `sim_real_cno.pth` (unzipped `submission_v3_cno.zip`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 82.645 | 69.734 | 90.362 | 22.625 | 50.521 | 63.178 | 8.5s |

### v4 + `sim_real_cno.pth` (unzipped `submission_v4_cno.zip`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 82.645 | 69.734 | 90.362 | 21.251 | 51.844 | 63.167 | 10.0s |

Same frozen point predictions as v3 (accuracy identical); SPS 51.8 vs 50.5 from calibrated intervals on 2/4 steps.

### v5 + `sim_real_cno.pth` (unzipped `submission_v5_cno.zip`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 82.645 | 69.734 | 90.362 | 22.894 | 50.862 | 63.300 | 8.3s |

### v11 + `sim_real_cno.pth` (staged `model.pth`, CPU)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 82.449 | 69.925 | 90.246 | 13.792 | 50.520 | 61.386 | 28.5s |

LoRA path end-to-end (36 Conv3d wrapped, 329,616 adapter params, 1-step SGD on adapters); CPU per-step is forward+backward bound, GPU will be ms-scale.

## Staged real-data runs (Kaggle GPU, `scripts/stage_real30.py` 30 traj / seed 42 — diagnostic, NOT leaderboard)

### v3 + `sim_real_cno.pth` on real-30 (2026-09-16)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 95.856 | 75.447 | 96.119 | 89.051 | 54.535 | 82.201 | 11ms |

Caveats: stats were fit on the same 30 trajectories (self-normalized, not official `mean_std_real.pt`); trajectories truncated to 200 frames. Still, the pattern matches Codabench: accuracy excellent, SPS (54.5, default ±5% band) the clear laggard → motivates v4 calibration.

### v6 width sweep on real-30 (2026-09-17, Kaggle GPU — probe notebook since removed, results kept)

Same frozen CNO, fixed normalized-space bands `predn ± w*|predn|`, official SPS (§4 == §5 exactly):

| half-width | total-width | sps_score | coverage | mean_nil |
|---|---|---|---|---|
| 0.050 | 0.100 | 54.33 | 0.2821 | 0.090 |
| 0.100 | 0.200 | 55.68 | 0.4202 | 0.181 |
| 0.200 | 0.400 | 56.50 | 0.5679 | 0.361 |
| 0.300 | 0.600 | 56.72 | 0.6548 | 0.542 |
| 0.500 | 1.000 | 56.81 | 0.7590 | 0.903 |

Monotone gains with fast decay (+1.35 / +0.82 / +0.22 / +0.09) — default ±5% far too narrow, peak at or beyond w=0.5. §5 reproduces §4 row-for-row on 270 steps (evaluator path confirmed); best w=0.5 packed. Same staged-data caveats as the v3 real-30 run above.

### v4 history/frames ablation on real-30 (2026-09-18, Kaggle GPU, notebook §5b — 270 steps)

| config | sps_score | coverage | mean_nil | steps-with-bounds |
|---|---|---|---|---|
| hist=2, frames=5 (shipped default) | 59.92 | 0.8211 | 0.425 | 270/270 |
| hist=1, frames=10 (recency probe) | 59.97 | 0.8102 | 0.404 | 270/270 |

Same 10 frames of residuals either way; recency is slightly sharper (nil 0.404 vs 0.425) at marginally lower coverage. Residuals are fairly stationary — consistent with the v7 alpha-flatness below. Configs otherwise: `sim_real_cno.pth`, `coverage: 0.90`, `fallback_frac: 0.05`. (Supersedes the earlier 59.97/0.7948/0.381 single-row report from the first §5b run.)

### v4 symmetry probe: symmetric q90 vs asymmetric [q05, q95] on real-30 (2026-09-19, Kaggle GPU, `notebook/eval_kaggle.ipynb`)

Same 270-step run, same predictions and table state (`table-mirror err 5.36e-07`); asymmetric band from signed-residual quantiles at the same nominal 90% level:

| band | sps_score | coverage | mean_nil |
|---|---|---|---|
| sym `q90(\|resid\|)` | 59.92 | 0.8211 | 0.425 |
| asym `[q05, q95]` | 59.91 | 0.8196 | 0.424 |

Per-channel tails: u `q95/|q05| = 1.130` (mild right skew), v `= 0.984`. No SPS difference — residuals are effectively symmetric at the 90% level, so keep v4's symmetric band.

### v12 FNO + frozen q90 band on real-30 (2026-09-19, Kaggle GPU, `notebook/eval_kaggle.ipynb` — 270 steps)

| rel_l2 | tke | mvpe | time | sps | final | per-step |
|---|---|---|---|---|---|---|
| 96.240 | 77.909 | 96.334 | 84.273 | 61.048 | 83.161 | 25.4ms |

SPS components (270/270 steps with bounds): raw 0.449313, coverage 0.7473, mean_nil 0.2347 (exp(-nil) 0.7969; branches dm 0.519 / tke 0.285 / mvpe 0.521).

Frozen beats online: the same FNO with v4's per-step band scored sps 60.53 at coverage 0.8180 / nil 0.374; freezing after 5 warmup windows drops coverage 7pts (0.747) but cuts width 37% (nil 0.235) — sharpness wins over coverage at these levels, net +0.52 sps. Step time also falls (25.4ms vs 30.6ms, no per-step quantile), final 83.16 vs 82.80 — best staged number. v12 now runs the frozen implementation (promoted from `submission_v13`, removed); the online-band numbers above are kept as history.

### v5 relative-q90 on real-30 (2026-09-18, Kaggle GPU, notebook §5c — 270 steps)

| sps_score | coverage | mean_nil | steps-with-bounds |
|---|---|---|---|
| 57.86 | 0.8169 | 0.843 | 270/270 |

Highest coverage of the frozen variants, but the sharpness tax eats it (`width ∝ |pred|` overcovers fast regions): trails v4 by ~2 sps. Relative scaling alone is too aggressive.

### v7 EMA alpha sweep on real-30 (2026-09-18, Kaggle GPU, notebook §5c — 270 steps)

| ema_alpha | sps_score | coverage | mean_nil | steps-with-bounds |
|---|---|---|---|---|
| 0.10 | 59.96 | 0.7949 | 0.383 | 270/270 |
| 0.30 | 59.97 | 0.7951 | 0.382 | 270/270 |
| 0.50 | 59.97 | 0.7950 | 0.382 | 270/270 |
| 0.70 | 59.97 | 0.7948 | 0.381 | 270/270 |
| 1.00 | 59.98 | 0.7944 | 0.381 | 270/270 |

Flat across the whole range (best 1.0, memoryless). The `ema_traj.png` trajectories overlap: global q90 is stable over time, so every alpha converges to the same band and only the 2–3-step transient differs. Memory doesn't matter here — v4's fixed window suffices; the variance is spatial, not temporal (see §7 below).

Full row (α=0.30 policy default; 270/270 steps with bounds, 207.8ms/step):

| variant | rel_l2 | tke | mvpe | time | sps | final |
|---|---|---|---|---|---|---|
| v7 (frozen, α=0.30) | 95.856 | 75.447 | 96.119 | 65.190 | 59.970 | 78.516 |

SPS components: raw 0.404208, coverage 0.7951, mean_nil 0.3821 (branches dm 0.472 / tke 0.242 / mvpe 0.478) — matches the α=0.30 sweep row exactly. Point scores match v3 to 3 decimals, confirming byte-identical frozen predictions. Step cost is the band, not the forward: 207.8ms vs v3's 11ms band-free (same-GPU caveat as always), which is why final (78.52) trails v12's (82.80) despite comparable SPS.

### v9 / v10 full rows on real-30 (2026-09-18, Kaggle GPU, notebook §4b/§5e — 270 steps)

| variant | rel_l2 | tke | mvpe | time | sps | final |
|---|---|---|---|---|---|---|
| v9 (1-step SGD, default band) | 95.29 | 75.62 | 95.54 | 50.48 | 53.93 | 74.17 |
| v10 (+ v4 q90 band) | 95.29 | 75.62 | 95.54 | 47.63 | 59.48 | 74.71 |

Point scores identical to 2 decimals — v9/v10 share byte-identical weight trajectories (same step, same pairs), as designed. The +5.55 sps / +0.54 final from v9 → v10 is pure calibration gain. v10 final 74.71 is the best staged number so far. Caveat: 1-step `1e-3` SGD barely moves point accuracy vs frozen (compare v3's 95.86/75.45/96.12) — adaptation's payoff here is letting the band ride fresher residuals, not better preds. Time cost of the backward pass: ~47–50 vs frozen-band runs (time subscore, Kaggle GPU).

### v8 speed-conditioned q90 on real-30 (2026-09-18, Kaggle GPU, notebook §5d — 270 steps)

| sps_score | coverage | mean_nil | steps-with-bounds |
|---|---|---|---|
| 58.72 | 0.7879 | 0.596 | 270/270 |

Trails v4 (59.92) despite conditioning: wider bands (nil 0.596 vs 0.425) yet lower coverage. The speed proxy + only 2 windows of table split across 10 bins doesn't capture the structure — per-bin quantile estimates are noisy and bin edges shift each step. Puzzle to fix, not a verdict on conditioning itself (see §7: the structure is spatial, speed is only its proxy).

### §7 spatial bias maps on real-30 (notebook §7, `bias_maps.png`)

`err = pred − target`, frozen CNO, raw space, averaged over 270 windows × time:

- **Bias ≪ RMSE.** Max |bias| ≈ 0.011 vs RMSE up to 0.03 (u): errors are variance-dominated, no correctable global offset. Width calibration is the game, not bias correction.
- **u bias is structured, v is not.** u overpredicts (red) near the body/lower-left (x 0–20, y 17–28) and underpredicts (blue) in the wake blob (x 35–60, y 15–25) plus a shear band (y ≈ 9–14). v bias is near-zero everywhere.
- **RMSE concentrates in the wake + shear layers** (u bright streak x > 20, y 10–25; v blob x ≈ 40, y ≈ 16); far field is dark. A global band wastes width on easy pixels and starves the wake — spatial (or speed-as-proxy) conditioning is the correct next lever.
- **u errors ~2× v errors** (RMSE scales 0.03 vs 0.015) — per-channel tables already handle this; keep them separate.
- Combined with the flat global-q90-over-time finding: residuals are temporally stationary but spatially heterogeneous → condition on space/speed, not on longer history.

## Leaderboard (Codabench, real `test_real`)
### `submission_v3_cno.zip` — v3 CNO, no adaptation (2026-09-16)

| rel_l2 | tke | mvpe | time | sps | final |
|---|---|---|---|---|---|
| 94.514 | 73.804 | 93.099 | 86.658 | 15.869 | **72.837** |

Frozen fine-tuned CNO is strong on point/probe accuracy (rel-L2 94.5, MVPE 93.1) and fast (time 86.7), but SPS 15.9 drags the mean — the default ±5%-magnitude interval is far too narrow for frozen point predictions. Takeaway: keep the CNO base, add calibrated uncertainty intervals (SPS fix) before spending budget on weight adaptation.
