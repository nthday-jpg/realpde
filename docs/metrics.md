# LTTTA Metrics

Leaderboard ranking uses `final_score`, the equal-weight mean of five 0-100
subscores (each clipped to `[0, 100]`). Higher is better for every score. This
page summarizes the Evaluation page; that page is authoritative.

- `rel_l2_score`: data fidelity from relative L2 error.
- `tke_score`: turbulent kinetic energy consistency.
- `mvpe_score`: mean velocity profile error at probe locations.
- `time_score`: per-step runtime efficiency.
- `sps_score`: safe prediction score from interval quality.

## Error Scores

Rel-L2, TKE, and MVPE are computed per evaluation window on the measured channels
`u, v` (pressure `p` is zero-filled and not scored), averaged over all windows,
and each maps to a 0-100 score via

```text
score = 100 / (1 + 0.5 * error)
```

- **Rel-L2** (data misfit): `||pred - target|| / ||target||` over the `u, v` field.
- **TKE**: relative L2 of the turbulent kinetic energy
  `0.5 * (var_t(u) + var_t(v))` (variance over the 20 time frames).
- **MVPE**: relative L2 of the time-averaged `u, v` at a fixed grid of wake probe
  points behind the airfoil.

## Time

Track 2 times **every `ttt_step` call**: adaptation on the previous pair,
controller decisions, optional LLM round-trips, and the forward prediction all
count. `reset_ttt_state` at trajectory boundaries and the evaluator's own
pre/post-processing do not. With `t_neural` the mean per-step wall time over the
whole stream and `t_numerical = 0.72896` s:

```text
r = t_neural / t_numerical
time_score = 100 * 1 / (1 + sqrt(r))
```

A missing, zero, negative, or non-finite per-step time scores 0.

## SPS

The Safe Prediction Score rewards predictions that are accurate *and* paired with
tight, well-calibrated intervals. It combines three physical error branches under
a coverage gate; the Evaluation page gives the exact formula. Summary:

- Each branch error `e` (DM = Rel-L2, TKE, MVPE) is squashed to `[0, 1)` by
  `pm = e / (0.5 + e)`.
- Per element, the reward is `(1 - pm) * exp(-(upper - lower) / sigma_global)`,
  counted only where the target lies inside `[lower, upper]`, then averaged.
- Branches combine with weights `DM 0.5 / TKE 0.3 / MVPE 0.2`, then map to
  `sps_score = 100 / (1 + exp(-weighted))`.

Interval width is normalized by the frozen constant `sigma_global = 0.0563870`
(mean of the `u, v` channel standard deviations on the official `train_real`
split, constant for the whole season). If you do not return `lower`/`upper`
arrays, the scorer uses a default band of `±5%` of `|prediction|`. Because the
weighted value is non-negative, `sps_score` lands in `[50, ~73]`.

## Composite

```text
final_score = mean(rel_l2_score, tke_score, mvpe_score, time_score, sps_score)
```

## Execution Time Limit

Separate from the per-step Time metric above, each submission must finish within
a wall-clock **10-minute** execution limit (Warm-up and Development phases,
container execution only; data download excluded). Submissions exceeding it are
marked Failed with no score.
