# Baseline Architectures and Checkpoints

The fine-tuned baseline checkpoints are stored in:

```text
data/baseline_checkpoints/sim_real_ft/
├── sim_real_cno.pth
├── sim_real_fno.pth
├── sim_real_fno_fp16.pth
└── sim_real_transolver.pth
```

All models consume and predict a complete normalized spatiotemporal volume:

```text
Input:  (B, 20, 32, 64, 3)  # time, height, width, (u, v, p)
Output: (B, 20, 32, 64, 3)
```

They predict all 20 output frames in one forward pass rather than autoregressively. Architecture reconstruction is defined in `load_baseline.py`; the training checkpoints themselves do not contain model configuration.

## Checkpoint summary

| Checkpoint | Architecture | Parameters | Training metadata |
|---|---|---:|---|
| `sim_real_cno.pth` | 3-level CNO/3D convolutional encoder-decoder | 7,960,467 | Iteration 5000; best iteration 5000; best validation loss 0.008612 |
| `sim_real_fno.pth` | 4-layer 3D Fourier Neural Operator | 50,357,955 logical parameter elements | Iteration 5000; best iteration 4800; best validation loss 0.008896 |
| `sim_real_fno_fp16.pth` | FP16-packed version of the same FNO | Same architecture | Training metadata omitted |
| `sim_real_transolver.pth` | 3-block structured-mesh Transolver | 12,541,259 | Iteration 5000; best iteration 4400; best validation loss 0.012241 |

All four checkpoints load successfully with `strict=True` through `load_baseline.py`.

## FNO3d

Effective configuration:

```python
FNO3d(
    modes1=4,
    modes2=12,
    modes3=16,
    n_layers=4,
    width=64,
)
```

Data flow:

```text
Input
  → append normalized (t, y, x) coordinates
  → Linear(6 → 64)
  → pad each volume dimension by 6
  → 4 Fourier operator blocks
  → crop padding
  → Linear(64 → 128 → 3)
  → Output
```

Each Fourier block combines a truncated 3D spectral convolution with a local `1×1×1` convolution, followed by BatchNorm. GELU is applied after every block except the last. The Fourier modes correspond to the temporal, height, and width dimensions.

`sim_real_fno_fp16.pth` contains the same network. Its 16 complex spectral tensors are stored as FP16 real/imaginary pairs and reconstructed when loaded.

## CNO3d

Effective configuration:

```python
CNO3d(
    in_dim=3,
    out_dim=3,
    in_size=64,
    N_layers=3,
    N_res=1,
    N_res_neck=6,
    channel_multiplier=32,
    latent_lift_proj_dim=64,
    activation="LeakyReLU",
    add_inv=True,
)
```

Channel structure:

```text
Input(3)
  → Lift: 3 → 64 → 16
  → Encoder: 16 → 32 → 64 → 128
  → Bottleneck: 6 residual blocks at 128 channels
  → Decoder: 128 → 64 → 32 → 16, with encoder skip concatenations
  → Projection: 32 → 64 → 3
  → Output
```

Each encoder level has one residual block. The network uses `3×3×3` convolutions, BatchNorm, LeakyReLU, and U-Net-style skip connections. Time is treated as the third convolutional dimension.

Although the classes describe CNO down/up blocks, the configured plain `LeakyReLU` path does not perform filtered resampling. Tensor resolution therefore remains `20×32×64` throughout; the implementation behaves like a resolution-preserving 3D convolutional U-Net.

## Transolver

Effective configuration:

```python
Model(
    space_dim=3,
    n_layers=3,
    n_hidden=256,
    n_head=8,
    fun_dim=0,
    out_dim=3,
    dropout=0.1,
    mlp_ratio=4,
    slice_num=16,
    H=64,
    W=32,
    D=20,
    Time_Input=False,
    unified_pos=False,
)
```

Data flow:

```text
Input volume
  → flatten to 40,960 points
  → preprocessing MLP: 3 → 512 → 256
  → add learned placeholder embedding
  → 3 Transolver blocks
  → LayerNorm + Linear(256 → 3)
  → reshape to the output volume
```

Each block contains:

1. LayerNorm, physics attention, and a residual connection.
2. LayerNorm, an MLP (`256 → 1024 → 256`), and a residual connection.

Physics attention avoids full attention across all 40,960 points. It uses `3×3×3` convolutions, splits features into 8 heads of 32 dimensions, softly aggregates points into 16 slice tokens per head, performs attention among those tokens, and projects the result back to every point.

The checkpoint does not encode `H`, `W`, or `D`; `load_baseline.py` derives them at runtime. Their product must match the number of flattened input points.

## Comparison

| Model | Main operation | Primary inductive bias |
|---|---|---|
| FNO | Truncated 3D Fourier convolutions | Global spectral interactions |
| CNO | Local 3D convolutions with skip connections | Local spatiotemporal structure |
| Transolver | Learned slice-token attention | Compressed global interactions |

## Implementation notes

- `src/realpde/rpde_baselines/model/load_model.py` assumes samples use `(T, H, W, C)` layout.
- CNO obtains `in_size` from `input_shape[2]`, which is `64` for the non-square `32×64` foil grid.
- FNO requires the output/input time ratio to be an integer, although this is not explicitly validated.
- Transolver hard-codes `Time_Input=False` and `unified_pos=False` in `load_model.py`.
- The training checkpoints contain model weights and loss metadata, but no optimizer state or architecture configuration.
