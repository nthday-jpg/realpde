# Agentic LTTTA Demo (bounded controller, offline-design / online-execution)

A submittable adaptation of the **agentic-LTTTA baseline**
(<https://github.com/PgUpDn/agentic_LTTTA>) to the official Track 2 protocol.
The upstream repository is the full method: a bounded online controller over a
fixed action space, whose policy knobs are tuned by an *offline* LLM design
layer, plus demos of *online* LLM action selection through an organizer
gateway. This folder shows how to run that idea under the competition contract.

## What differs from the upstream demos

1. **Prev-target stream.** Official evaluation reveals only the ground truth of
   the *previous* window at each step. The controller therefore scores its own
   previous prediction against that revealed target (rel-L2) and adapts on the
   genuine previous (input, target) pair; the current target is never seen.
2. **Gateway via environment.** The evaluation container injects
   `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `REALPDE_GATEWAY_MODEL`. With
   `mode: llm` the controller sends a compact numeric state to the gateway
   (OpenAI-compatible, single turn) and receives one bounded action. Missing
   env, timeouts, or malformed replies fall back to the rule policy: the LLM
   can never crash the run. Direct participant API keys are prohibited.
3. **Everything in `ttt_step` is timed**, LLM round-trips included. The policy
   budgets them (`llm_every`, `llm_timeout_s`, `time_budget_s`).

## Modes

- `mode: rule` (default): deterministic online execution; `policy.yaml` is the
  offline-design artifact. Tune it by hand or with the upstream ADK design
  agents and ship the tuned file.
- `mode: llm`: online agentic execution through the organizer gateway, same
  bounded action space (`skip_update` / `recalibrate` / `update_adapter`).
- `mode: fixed`: ablation (e.g. `fixed_action: skip_update` = no adaptation).

## Run it

```bash
# from the starting_kit root: rule mode, CPU
python local_eval.py --submission agentic_demo

# online mode is exercised automatically in the official container when
# mode: llm is set; locally you can point OPENAI_* at any OpenAI-compatible
# endpoint to try it.
```

To use a real base model instead of the built-in tiny net, copy
`load_baseline.py` and `rpde_baselines/` from the kit into your submission zip
and place a baseline checkpoint as `model.pth` (see the kit README "Baseline
Models"; mind the 256 MB cap -- `sim_real_cno.pth` at 32 MB is a good start).
