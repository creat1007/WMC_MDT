# WMC_MDT — Radar Echo Nowcasting Model

A radar extrapolation nowcasting system based on [NowcastNet](https://www.nature.com/articles/s41586-023-06184-4),
adapted and trained for operational use over China (North China / South China).

**Input: past 20 frames (2 hours) → Output: next 30 frames (3 hours), at 6-minute intervals.**

---

## Features

- **Physics-generative hybrid architecture**: an Evolution network learns optical-flow advection, while a Generative network (with a discriminator) synthesizes fine-scale detail
- **Adversarial training (GAN)**: spatiotemporal discriminator + pooling regularization, producing sharp fields with realistic convective cores instead of the over-smoothed output typical of L1 regression
- **Frequency-balanced loss**: foreground and background are averaged separately, preventing the model collapse caused by radar fields being ~99% zero background
- **Multi-region joint training**: North + South China data can be mixed into a single model
- **Operational-ready**: single-level (composite reflectivity) and multi-level (12 height levels) pipelines, suitable for scheduled real-time runs

---

## Quick Start

### Environment

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py \
    --train_data_path /path/to/data_regionA,/path/to/data_regionB \
    --data_max 55.5 \
    --gan --lambda_adv 0.03 --lambda_pool 0.5 \
    --lambda_motion 0.0003 --lambda_evo 3.0 \
    --warp_mode bilinear \
    --lr 1e-4 --batch_size 12 \
    --save_dir ./checkpoints
```

Multi-GPU: `torchrun --nproc_per_node=4 train.py ...`

**Warm start from an existing model:**

```bash
python train.py --pretrained_model ./checkpoints/best_model.ckpt --data_max 55.5 ...
```

> ⚠️ When warm-starting on a different dataset, `--data_max` **must** match the pretrained value.
> Changing the normalization scale invalidates the learned weights.

### Inference

```bash
python run.py \
    --forecast_only \
    --dataset_path /path/to/input_dir \
    --pretrained_model ./checkpoints/best_model.ckpt \
    --gen_frm_dir ./results \
    --input_length 20 --total_length 50 \
    --img_height 512 --img_width 512 \
    --warp_mode bilinear
```

### Data Preprocessing (optional, large speedup)

Convert NetCDF into `.npy` caches for memory-mapped reads during training:

```bash
python preprocess_cache.py --data_path /path/to/nc_dir --cache_dir /path/to/cache
```

Convert colored radar images (PNG) into dBZ training data:

```bash
python preprocess_cache.py --input_type image \
    --data_path /path/to/png_dir --cache_dir /path/to/npy \
    --pal_file /path/to/palette.txt
```

---

## Repository Layout

```
├── train.py                    # Training (GAN / warm start / multi-GPU)
├── run.py                      # Inference
├── preprocess_cache.py         # Preprocessing (NetCDF / colored images → npy)
├── revive_flow.py              # Fix for optical-flow layer weight collapse (see below)
├── diagnose_typhoon.py         # Diagnostic: isolate structure loss in advection vs. generation
└── nowcasting/
    ├── models/nowcastnet.py            # Main model
    ├── layers/evolution/               # Evolution network (optical flow + intensity)
    ├── layers/generation/              # Generative network
    │   └── discriminator.py            # Spatiotemporal discriminator + hinge loss + pooling reg.
    └── data_provider/                  # Data loading
```

---

## Training Notes

### Key Hyperparameters

| Argument | Description | Suggested |
|---|---|---|
| `--data_max` | Normalization reference (dBZ). Must match the pretrained value when warm-starting | `55.5` |
| `--lambda_adv` | Adversarial loss weight. Too high destabilizes training; too low gives no sharpening | `0.03` |
| `--lambda_pool` | Pooling regularization; constrains coarse-scale rainfall so the discriminator cannot push the model to fabricate strong echoes | `0.5` |
| `--lambda_evo` | Evolution self-supervision weight | `3.0` |
| `--lambda_motion` | Optical-flow smoothness regularization. Too large suppresses high-gradient flow such as rotation | `0.0003` |
| `--warp_mode` | **Must be `bilinear`** (see below) | `bilinear` |

### Evaluation

**CSI** (at 20/30/40 dBZ) is computed on a held-out validation set every `--val_interval` epochs.

> **Note**: under GAN training, CSI may drop slightly while visual quality clearly improves.
> CSI penalizes sharp cores that are slightly displaced, yet rewards blurry mean fields.
> Judge model quality primarily by **case visualization**, with CSI as a secondary indicator.

---

## Issues Found During Development

A few high-impact problems identified while debugging this system, documented for reference.

### 1. Model collapse to blank output

Under a plain L1 loss, radar fields are ~99% zero background, so predicting all-zeros drives the
loss down to ~1e-5 — a false convergence where the loss keeps decreasing but CSI stays at 0.

**Fix**: `balanced_l1` — foreground (≥15 dBZ) and background are averaged *separately* and then
combined, so foreground error is never diluted by the vast background. A validity mask and
dBZ-graded weighting were added as well.

### 2. Over-smoothed forecasts and intensity collapse

L1 regression favors a "safe" mean field: convective cores are flattened and echoes spread into
broad weak regions.

**Fix**: reintroduce NowcastNet's adversarial training (spatiotemporal discriminator + hinge loss
+ pooling regularization). Warm-starting the adversarial phase from an L1-pretrained model is far
more stable than training a GAN from scratch.

### 3. Optical-flow layer weight collapse (`revive_flow.py`)

**Symptom**: echoes barely moved; structure was smeared out within an hour. Regardless of which
loss weight was tuned, the predicted flow field came out *bit-for-bit identical*.

**Root cause**: `warp` used `mode="nearest"`, while the model's predicted displacement was only
~0.02–0.2 px/frame. Nearest-neighbor rounding truncated that to zero, so the flow had no effect on
the output → gradient ≈ 0 → the `outc_v` weights collapsed to std ≈ 1e-4 → the flow degenerated
into a constant field and could never recover (a self-locking loop).

**Fix**:
1. switch `warp` to `bilinear` (sub-pixel capable and differentiable);
2. `revive_flow.py` re-initializes the collapsed `outc_v` layer and clears optimizer state, then warm-start.

After the fix, flow magnitude grew ~15×, restoring genuine advection.

```bash
python revive_flow.py --ckpt old.ckpt --out revived.ckpt
python train.py --pretrained_model revived.ckpt --warp_mode bilinear --lr 1e-4 ...
```

---

## Known Limitations

- **Typhoons and strongly rotating systems**: spiral structure is still hard to maintain beyond
  2–3 hours. Flow magnitude is fixed, but a single U-Net struggles to learn the coherent rotational
  field needed to sustain a vortex. Usable within ~1 hour.
- **Convective initiation**: pure radar extrapolation cannot predict newly developing convection —
  an inherent limitation of this class of methods.
- **Underestimated propagation speed**: acceleration and track changes carry no signal in past
  frames, so extrapolation models tend to lag.

Planned direction: incorporate NWP wind fields to guide the Evolution network.

---

## Acknowledgements

- Architecture based on [NowcastNet](https://github.com/thuml/NowcastNet) (Zhang et al., *Nature* 2023)

## License

See [LICENSE](LICENSE).
