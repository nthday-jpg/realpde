# LTTTA Submission Interface

Your `submission.py` exposes a factory function and returns a test-time-adapting
model. Evaluation is **duck-typed**: only the method names below are checked, so
subclassing `ttt_model.py`'s `TTTModel` is optional.

## Factory Function

```python
def get_ttt_model(submission_dir: str, device: str):
    """Return an object implementing the TTT interface below."""
```

Called once, before the stream starts. Construction and checkpoint loading are
**not timed**. `submission_dir` is the extracted archive root (load
`model.pth` from there with relative paths); `device` is `"cuda"` on the
platform.

## Model Interface

```python
def reset_ttt_state(self) -> None:
    # Called at the start of every trajectory. Restore checkpoint weights and
    # clear adaptation state. NOT timed.

def ttt_step(self, input_norm, prev_target_norm=None):
    # Called once per step, batch size 1, in normalized space. FULLY timed.
    #   input_norm       : (1, 20, 32, 64, 3) current input window.
    #   prev_target_norm : (1, 20, 32, 64, 3) ground truth of the PREVIOUS step,
    #                      or None on the first step of a trajectory.
    # Returns (pred_norm, info):
    #   pred_norm : (1, 20, 32, 64, 3) prediction for the current window.
    #   info      : dict. The evaluator reads ONLY info["adapt_loss"] (a python
    #               float, or None when no adaptation happened this step); any
    #               other keys you add are ignored.
```

The raw `64 x 128` fields are evaluated at a `2x` down-sampled `32 x 64`
resolution, so every tensor has shape `(1, 20, 32, 64, 3)`.

## Streaming Protocol

Trajectories are revealed strictly in time order, batch size 1. Window `k`'s
target frames are window `k+1`'s input frames (20-in to 20-out, stride 20).
Trajectory boundaries trigger `reset_ttt_state()`.

The canonical `ttt_step` structure:

1. If `prev_target_norm is not None`, adapt using your **cached previous input**
   and this revealed **previous target** (a genuine `input -> target` pair).
   Do not pair the previous target with the current input.
2. Predict the current `input_norm` without using the current target.
3. Cache the current input for the next step.

The evaluator caches the `(input, target)` pair only *after* the timed
`ttt_step` returns, so the current step's target is never available to the
current prediction. See `submission_template.py` for a complete reference
implementation.

## Channel Convention

Channels are ordered `[u, v, p]`. Real-world scoring uses measured channels `u`
and `v`; pressure `p` is zero-filled and can be returned as zeros. The evaluator
handles normalization and denormalization, so work entirely in normalized space.

## Environment and Limits

- Extracted archive (checkpoint included) must stay under **256 MB**; a
  complex-safe fp16 packing tool ships as `pack_ckpt_fp16.py`.
- Evaluation runs offline in `w3nhao/realpde-track2:v1` (PyTorch 2.2.2, CUDA
  12.1, Python 3.10). Nothing is installed at evaluation time; vendor pure-Python
  dependencies inside your archive.
- Optional LLM calls must go through the organizer gateway (see the FAQ); their
  latency counts toward the Time metric.
- Your model runs in an **isolated subprocess**. It reads your submission archive
  and the Python runtime normally, writes freely (temp files, caches,
  checkpoints), and uses torch / CUDA / numpy as usual. Only the hidden
  evaluation data (ground truth) is unreadable from disk: any attempt to open it
  is denied by the kernel. You never need it, because `ttt_step` is handed the
  normalized input and previous target directly.
