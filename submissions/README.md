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

## Changelog

- _Fill after each eval, e.g._ `v1 + sim_real_cno.pth: rel_l2_score 61.2, time 12ms, 33MB — submitted as v1.zip`
