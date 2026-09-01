# RealPDE Track 2 LTTTA Starting Kit

This kit contains the minimal files to build a Codabench submission for Track 2:
Long-Term Test-Time Adaptation on streaming real-world PIV data.

## Quick Start

### Run baseline with TTT adaptation

```python
from models import ModelAdapter
from submission_template import ReferenceTTTModel

adapter = ModelAdapter.from_baseline("data/baseline_checkpoints/sim_real_ft/sim_real_cno.pth", device="cuda")
model = ReferenceTTTModel(adapter.model, device="cuda")
```

### Run pretrained model with TTT adaptation

```python
from models import ModelAdapter, get_model
from submission_template import ReferenceTTTModel

model = get_model("unet")
adapter = ModelAdapter.from_pretrained(model, "checkpoints/best.pth", device="cuda")
model = ReferenceTTTModel(adapter.model, device="cuda")
```

### Pretrain on Kaggle

1. Upload data as a Kaggle dataset named `realpde`
2. Open `notebook/pretrain_kaggle.ipynb`
3. Edit config variables (DATA_PATH, MODEL_NAME, etc.)
4. Run all cells

### Pretrain locally

```bash
accelerate launch trainer.py
```

Set config via environment variables or notebook globals before launch.

---

## Project Structure

```
realpde/
├── models/
│   ├── __init__.py          # get_model() factory + ModelAdapter
│   ├── base.py              # PretrainModel ABC
│   ├── adapter.py           # ModelAdapter: wraps baselines or pretrained models
│   └── unet/
│       ├── __init__.py
│       ├── config.py        # UNetConfig dataclass
│       └── model.py         # Per-frame 2D UNet
│
├── datasets/
│   ├── __init__.py
│   └── pde_dataset.py       # HDF5 loader, (input, target) windows
│
├── trainer.py               # Accelerate pretrainer (teacher forcing, wandb)
├── notebook/
│   └── pretrain_kaggle.ipynb
│
├── submission.py            # Reference TTT submission (TinyForecaster + 1-step SGD)
├── submission_template.py   # Template for your submission
├── ttt_model.py             # TTTModel base class
├── load_baseline.py         # Load FNO/CNO/Transolver checkpoints
├── pack_ckpt_fp16.py        # Pack fp32 checkpoint to fp16 (256 MB cap)
├── scoring.py               # Official leaderboard scoring
├── local_eval.py            # Local CPU smoke test
└── rpde_baselines/          # Vendored baseline model code
```

---

## Interface

All models use the same tensor convention: `(B, T, H, W, C)` where `C = 3` (u, v, p).

| Constant | Value |
|----------|-------|
| `IN_STEP` | 20 |
| `OUT_STEP` | 20 |
| `HEIGHT` | 32 |
| `WIDTH` | 64 |
| `CHANNELS` | 3 |

### `submission.py` entry point

```python
def get_ttt_model(submission_dir, device):
    ...  # return an object with reset_ttt_state() and ttt_step()
```

- `reset_ttt_state()`: restore checkpoint weights, clear adaptation state (not timed)
- `ttt_step(input_norm, prev_target_norm=None)`: adapt + predict, returns `(pred_norm, info)` where `info["adapt_loss"]` is read by the evaluator

---

## Adding a New Model

### 1. Create the model folder

```
models/
└── your_model/
    ├── __init__.py
    ├── config.py        # config dataclass
    └── model.py         # nn.Module inheriting PretrainModel
```

### 2. Define the config dataclass

```python
# models/your_model/config.py
from dataclasses import dataclass

@dataclass
class YourModelConfig:
    in_step: int = 20
    out_step: int = 20
    in_channels: int = 3
    out_channels: int = 3
    # your hyperparameters here
    hidden_dim: int = 64
    n_layers: int = 4
```

### 3. Implement the model

```python
# models/your_model/model.py
import torch
import torch.nn as nn
from models.base import PretrainModel
from models.your_model.config import YourModelConfig

class YourModel(PretrainModel):
    def __init__(self, config: YourModelConfig | None = None):
        super().__init__()
        if config is None:
            config = YourModelConfig()
        self.config = config
        # build layers here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, H, W, C) -> output: (B, T_out, H, W, C)
        ...
        return output
```

### 4. Register in the factory

```python
# models/__init__.py
from models.your_model import YourModel

_MODEL_REGISTRY = {
    "unet": UNet,
    "your_model": YourModel,  # add here
}
```

### 5. Add config to trainer.py

In `trainer.py`, add an `elif` branch in the model construction block:

```python
elif model_name == "your_model":
    from models.your_model import YourModel, YourModelConfig
    model_cfg = YourModelConfig(
        in_step=in_step, out_step=out_step,
        # pass config params
    )
    model = YourModel(model_cfg)
```

### 6. Use it

```python
from models import get_model, ModelAdapter

model = get_model("your_model", hidden_dim=128, n_layers=6)
adapter = ModelAdapter.from_pretrained(model, "checkpoints/best.pth")
```

---

## Training Data Format

`.h5` files with flat datasets:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `u` | `(T, 64, 128)` | float16 | x-velocity |
| `v` | `(T, 64, 128)` | float16 | y-velocity |
| `p` | `(T, 64, 128)` | float16 | pressure (zeros for real data) |

`PDEDataset` loads these, applies spatial subsampling (`::2` to get `32x64`), and creates `(input_window, target_window)` pairs.

---

## ModelAdapter

`ModelAdapter` wraps any model (baseline or pretrained) to a uniform interface:

```python
# From baseline checkpoint (auto-detects FNO/CNO/Transolver)
adapter = ModelAdapter.from_baseline("sim_real_cno.pth", device="cuda")

# From pretrained model + checkpoint
adapter = ModelAdapter.from_pretrained(YourModel(config), "best.pth", device="cuda")

# Same forward signature regardless of origin
pred = adapter(input_tensor)  # (B, 20, 32, 64, 3) -> (B, 20, 32, 64, 3)

# Reset to initial weights
adapter.reset_weights()
```

---

## Trainer Config

`trainer.py` reads config from notebook globals or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `data/train_sim` | Path to `.h5` files |
| `MODEL_NAME` | `unet` | Model name |
| `IN_STEP` | 20 | Input frames |
| `OUT_STEP` | 20 | Output frames |
| `INTERVAL` | 20 | Temporal stride |
| `LR` | 1e-3 | Learning rate |
| `EPOCHS` | 50 | Training epochs |
| `BATCH_SIZE` | 8 | Local batch size |
| `SUB_S` | 2 | Spatial subsample |
| `SAVE_DIR` | `checkpoints` | Checkpoint output dir |
| `WANDB_PROJECT` | `realpde-pretrain` | Wandb project |
| `UNET_CHANNELS` | 16 | UNet base channels |
| `UNET_N_LAYERS` | 2 | UNet encoder depth |

---

## Baseline Models

Checkpoints in `data/baseline_checkpoints/`:

```
sim_pretrain/          sim_real_ft/           (fine-tuned on real, best start)
├── sim_cno.pth        ├── sim_real_cno.pth      32 MB
├── sim_fno.pth        ├── sim_real_fno.pth      384 MB (fp32)
├── sim_fno_fp16.pth   ├── sim_real_fno_fp16.pth 192 MB (fp16-packed)
└── sim_transolver.pth └── sim_real_transolver.pth 50 MB
```

Load any baseline:

```python
from models import ModelAdapter
adapter = ModelAdapter.from_baseline("sim_real_cno.pth")
```

---

## Submission Size

Extracted archive must stay under **256 MB**. Pack large checkpoints:

```bash
python pack_ckpt_fp16.py model_fp32.pth model.pth
```

---

## Local Smoke Test

```bash
python local_eval.py --submission .
```

Mirrors the official streaming loop on bundled `example_data/`, reports subscores.
