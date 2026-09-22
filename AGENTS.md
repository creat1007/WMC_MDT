# AGENTS.md — WMC_MDT

Local rules for agents working in this repository.

## What this repo is

Radar echo nowcasting (0–3 h) for China, built on the NowcastNet architecture.
**20 input frames (2 h) → 30 forecast frames (3 h), 6-minute interval, 512×512.**

This is an **observation-driven regional nowcasting** model. It is *not* a reanalysis-driven
global model, so the conventions here differ from `xmetai-core`:

| | `xmetai-core` | this repo |
|---|---|---|
| Data | Zarr (ERA5 / CRA / ORAS5 / S2S) | radar mosaic `.bin` → `.nc` |
| Config | LazyConfig | `argparse` in `run.py` / `train.py` |
| Model API | `base_predictor`, `forward(data)->loss_dict`, `test_cascade(data)->outputs` | `Model` + `model_factory` |
| Cadence | 6 h / 24 h | 6 min |

Do not refactor this repo toward the LazyConfig / `base_predictor` contract unless that
migration is explicitly requested — it is a substantial rewrite, not a cleanup.

## Layout

```
nowcasting/
├── models/nowcastnet.py         # main model
├── models/model_factory.py      # Model wrapper: train/test entry points
├── layers/evolution/            # evolution network (optical flow + intensity)
├── layers/generation/           # generative network
│   └── discriminator.py         # spatiotemporal discriminator, hinge loss, pooling reg.
└── data_provider/               # loaders (loader.py = inference, train_loader.py = training)
run.py                           # inference entry
train.py                         # training entry
revive_flow.py                   # repair tool for collapsed optical-flow weights
diagnose_typhoon.py              # evolution-vs-generation stage diagnostics
preprocess_cache.py              # dataset caching
```

## Non-negotiable rules

- **Never commit data or weights.** `training_data/`, `*.ckpt`, `*.pth`, `*.nc`, `*.npy`,
  `results/`, `cache/` are gitignored. Check `git status` before committing.
- **No internal paths.** Use `/path/to/...` placeholders in code, comments and docs.
  Real cluster paths must not enter the repository.
- **English only** in code, comments, docstrings, log messages and docs.
- **Do not claim training / export / deployment was validated unless it actually ran.**

## Domain facts an agent must not re-derive incorrectly

- **`data_max` must match between training and inference.** It is read from the checkpoint.
  The current models use `55.5` (p99.9 of the training distribution, in dBZ). A mismatch
  silently degrades output rather than erroring.
- **`warp` must use `mode="bilinear"`, never `"nearest"`.** With `nearest`, sub-pixel
  displacements round to zero, the optical-flow branch stops receiving gradient, and
  `outc_v` collapses to a constant field — a self-locking failure that no loss reweighting
  can escape. `revive_flow.py` exists to recover from that state.
- **`balanced_l1` averages foreground (≥15 dBZ) and background separately, then adds them.**
  Any `sum / total_pixels` normalization dilutes the foreground gradient until the model
  collapses to all-zeros — radar fields are ~99% zero background.
- **The discriminator must stay in float32.** AMP + GAN triggers an
  "unscale FP16 gradients" conflict.
- **CSI is not a sufficient quality criterion.** It rewards blurry mean fields and penalizes
  sharp cores that are slightly displaced, so adversarial training can lower CSI while
  clearly improving the forecast. Judge model quality primarily by case visualizations.

## Validation posture

Use the narrowest meaningful check: a focused unit test, a config/arg sanity check, a
read-only data schema inspection, or a line-by-line tensor-shape trace. Report the command
that actually ran and its output.

## Git

Do not commit, push, open PRs, or change remotes unless explicitly asked.
