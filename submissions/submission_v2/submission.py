"""Agentic LTTTA demo submission for RealPDE Track 2.

A self-contained, submittable adaptation of the agentic-LTTTA baseline
(https://github.com/PgUpDn/agentic_LTTTA) to the official Track 2 protocol:

  * offline-design / online-execution split: the bounded controller reads its
    knobs from ``policy.yaml`` (tune it offline, e.g. with the upstream repo's
    ADK design agents); the scored loop stays deterministic under
    ``mode: rule``.
  * adaptation signal is the official prev-target stream: at step t the
    evaluator reveals the ground truth of step t-1 only.
  * optional online LLM action selection (``mode: llm``) goes through the
    organizer gateway. The evaluation container injects ``OPENAI_BASE_URL``,
    ``OPENAI_API_KEY`` and ``REALPDE_GATEWAY_MODEL``; no participant keys.
    If the env is absent or any call fails, the controller degrades to the
    rule policy -- the submission never crashes because of the LLM.

Controller state -> bounded action, once per step:

  state  = {prev-pair rel-L2, error EMA slope, step index, time budget left}
  action in {skip_update, recalibrate, update_adapter}

``update_adapter`` runs a few SGD steps on the revealed previous (input,
target) pair (scope ``bn`` = norm-layer parameters only, or ``all``);
``recalibrate`` maintains a cheap additive output correction; ``skip_update``
does nothing. Everything inside ``ttt_step`` counts toward the Time metric,
including LLM latency, so the policy budgets both.

Files expected next to this one: ``policy.yaml`` (knobs) and optionally
``model.pth`` (a real base checkpoint; see the kit's ``load_baseline.py`` --
ship ``rpde_baselines/`` in your zip to use CNO/FNO/Transolver). Without a
checkpoint a tiny residual conv net keeps the demo runnable on CPU.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:  # optional convenience base class (duck-typed evaluation)
    from ttt_model import TTTModel
except Exception:  # pragma: no cover
    TTTModel = nn.Module  # type: ignore

IN_STEP = 20
CHANNELS = 3
ACTIONS = ("skip_update", "recalibrate", "update_adapter")

DEFAULT_POLICY: Dict[str, Any] = {
    "mode": "rule",            # rule | llm | fixed
    "fixed_action": "skip_update",
    "base_model": "",          # "" (auto/tiny) | cno | fno | transolver
    "err_low": 0.035,          # below: prediction fine, skip
    "err_high": 0.11,          # above: adapt weights
    "ema_beta": 0.7,
    "adapt_steps": 5,
    "adapt_lr": 1e-3,
    "adapt_scope": "bn",       # bn | all
    "calib_momentum": 0.5,
    "time_budget_s": 60.0,     # per-trajectory ttt_step wall-clock budget
    "llm_every": 5,            # ask the LLM every k-th step (latency control)
    "llm_timeout_s": 5.0,
    "llm_max_tokens": 128,
}


def _load_policy(path: str) -> Dict[str, Any]:
    """Read policy.yaml; tolerate a missing PyYAML with a flat k:v parser."""
    policy = dict(DEFAULT_POLICY)
    if not os.path.exists(path):
        return policy
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception:
        loaded = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if ":" not in line:
                    continue
                k, v = (s.strip() for s in line.split(":", 1))
                if v.lower() in ("true", "false"):
                    loaded[k] = v.lower() == "true"
                else:
                    try:
                        loaded[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                    except ValueError:
                        loaded[k] = v
    policy.update({k: v for k, v in loaded.items() if k in policy})
    return policy


class TinyForecaster(nn.Module):
    """Fallback base model (same as the template): near-identity residual conv."""

    def __init__(self, channels: int = CHANNELS, hidden: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, hidden)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        z = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        z = z + self.conv2(torch.relu(self.norm(self.conv1(z))))
        return z.permute(0, 2, 3, 1).reshape(b, t, h, w, c)


def _rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target).clamp_min(1e-8)
    return float((num / den).detach().cpu())


class _LLMActionPicker:
    """Single-turn bounded action selection through the organizer gateway.

    Uses the OpenAI-compatible client against ``OPENAI_BASE_URL`` with
    ``REALPDE_GATEWAY_MODEL``. Any failure (missing env/package, timeout,
    malformed reply, off-menu action) returns None and the caller falls back
    to the rule policy. The state summary contains numbers only -- no data,
    no ground truth.
    """

    def __init__(self, timeout_s: float, max_tokens: int):
        self.model = os.environ.get("REALPDE_GATEWAY_MODEL", "")
        self.available = bool(
            os.environ.get("OPENAI_BASE_URL") and os.environ.get("OPENAI_API_KEY")
            and self.model
        )
        self._client = None
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def pick(self, state: Dict[str, float], allowed: List[str]) -> Optional[str]:
        if not self.available:
            return None
        try:
            if self._client is None:
                from openai import OpenAI  # shipped in the Track 2 image
                self._client = OpenAI(timeout=self.timeout_s)
            prompt = (
                "You control test-time adaptation of a flow forecaster. "
                "State: " + json.dumps(state) + ". "
                "Reply with JSON {\"action\": <one of " + json.dumps(allowed) + ">}."
            )
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            start, end = text.find("{"), text.rfind("}")
            action = json.loads(text[start:end + 1]).get("action")
            return action if action in allowed else None
        except Exception:
            return None


class AgenticTTTModel(TTTModel):
    """Bounded-controller TTT model implementing the official contract."""

    def __init__(self, base: nn.Module, device: str, policy: Dict[str, Any]):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.policy = policy
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._llm = _LLMActionPicker(policy["llm_timeout_s"], policy["llm_max_tokens"])
        self.reset_ttt_state()

    # ---- adaptation parameter scope ------------------------------------
    def _adapt_params(self):
        scope = str(self.policy["adapt_scope"]).lower()
        if scope == "bn":
            named = [p for n, p in self.base.named_parameters()
                     if ("norm" in n.lower() or "bn" in n.lower()) and p.requires_grad]
            if named:
                return named
        return [p for p in self.base.parameters() if p.requires_grad]

    # ---- contract -------------------------------------------------------
    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))
        self._opt = torch.optim.SGD(self._adapt_params(), lr=float(self.policy["adapt_lr"]))
        self._prev_input: Optional[torch.Tensor] = None
        self._prev_pred: Optional[torch.Tensor] = None
        self._calib = torch.zeros(1, device=self.device)
        self._err_ema: Optional[float] = None
        self._err_prev_ema: Optional[float] = None
        self._step = 0
        self._spent_s = 0.0

    def _decide(self, err: float, slope: float, budget_left: float) -> Tuple[str, str]:
        p = self.policy
        rule = ("update_adapter" if err > p["err_high"]
                else "skip_update" if err < p["err_low"]
                else "recalibrate")
        mode = str(p["mode"]).lower()
        if mode == "fixed":
            return str(p["fixed_action"]), "fixed"
        if mode == "llm" and self._step % max(int(p["llm_every"]), 1) == 0 \
                and budget_left > 2.0 * p["llm_timeout_s"]:
            state = {"rel_l2": round(err, 4), "slope": round(slope, 4),
                     "step": self._step, "budget_left_s": round(budget_left, 1)}
            picked = self._llm.pick(state, list(ACTIONS))
            if picked is not None:
                return picked, "llm"
        return rule, "rule"

    def _apply(self, action: str, prev_target: torch.Tensor) -> Optional[float]:
        if action == "update_adapter" and self._prev_input is not None:
            self.base.train()
            loss_val = None
            for _ in range(max(int(self.policy["adapt_steps"]), 1)):
                self._opt.zero_grad()
                loss = torch.mean((self.base(self._prev_input) - prev_target) ** 2)
                loss.backward()
                self._opt.step()
                loss_val = float(loss.detach().cpu())
            return loss_val
        if action == "recalibrate" and self._prev_pred is not None:
            m = float(self.policy["calib_momentum"])
            bias = (prev_target - self._prev_pred).mean()
            self._calib = (1.0 - m) * self._calib + m * bias
            return float(torch.mean((self._prev_pred - prev_target) ** 2).cpu())
        return None

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        t0 = time.perf_counter()
        input_norm = torch.as_tensor(input_norm).to(self.device)

        action, source, adapt_loss = "skip_update", "first_step", None
        if prev_target_norm is not None and self._prev_pred is not None:
            prev_target_norm = torch.as_tensor(prev_target_norm).to(self.device)
            err = _rel_l2(self._prev_pred, prev_target_norm)
            beta = float(self.policy["ema_beta"])
            self._err_prev_ema = self._err_ema
            self._err_ema = err if self._err_ema is None else beta * self._err_ema + (1 - beta) * err
            slope = 0.0 if self._err_prev_ema is None else self._err_ema - self._err_prev_ema
            budget_left = float(self.policy["time_budget_s"]) - self._spent_s
            if budget_left <= 0.5:
                action, source = "skip_update", "budget_guard"
            else:
                action, source = self._decide(err, slope, budget_left)
            adapt_loss = self._apply(action, prev_target_norm)

        self.base.eval()
        with torch.no_grad():
            pred_norm = self.base(input_norm) + self._calib

        self._prev_input = input_norm.detach()
        self._prev_pred = pred_norm.detach()
        self._step += 1
        self._spent_s += time.perf_counter() - t0
        # The evaluator reads only info["adapt_loss"]; `action` / `source` are
        # local-debug fields (log them yourself if useful) and are ignored on the
        # platform, so the returned info stays minimal.
        return pred_norm, {"adapt_loss": adapt_loss}


def _build_base(submission_dir: str, device: str, policy: Dict[str, Any]) -> nn.Module:
    """Prefer a real baseline checkpoint (kit load_baseline + rpde_baselines in
    the zip); otherwise the runnable TinyForecaster fallback. Set
    ``base_model: cno|fno|transolver`` in policy.yaml when the checkpoint
    filename does not carry the architecture name."""
    ckpt = os.path.join(submission_dir, "model.pth")
    if os.path.exists(ckpt):
        try:
            import sys
            if submission_dir not in sys.path:
                sys.path.insert(0, submission_dir)
            from load_baseline import load_baseline  # type: ignore
            mt = str(policy.get("base_model") or "").strip().lower()
            if mt in ("cno", "fno", "transolver"):
                base, _meta = load_baseline(mt, ckpt, device=device)
            else:
                base, _meta = load_baseline(ckpt, device=device)
            return base
        except Exception:
            base = TinyForecaster()
            state = torch.load(ckpt, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            base.load_state_dict(state)
            return base
    return TinyForecaster()


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    policy = _load_policy(os.path.join(submission_dir, "policy.yaml"))
    base = _build_base(submission_dir, device, policy)
    return AgenticTTTModel(base, device, policy)
