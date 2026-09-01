#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose typhoon structure loss: determine whether structure is lost in the evolution
(advection) stage or in the generation stage.

The model has two stages:
  1. evolution network  -> evo_result (pure advection + intensity, no GAN involved)
  2. generative network -> gen_result (final output, refined by the GAN)
This script runs one forward pass, extracts both, and reports structure metrics per lead
time along with a comparison figure.

Usage (on a machine that has the checkpoint and the input NetCDF):
    python diagnose_typhoon.py \
        --ckpt   /path/to/checkpoint_epoch_25.ckpt \
        --input  /path/to/typhoon_case_input.nc \
        --out_dir ./diag_typhoon

How to read the results:
  - evo keeps structure but gen is blurry  -> the problem is in the generative network / GAN
  - evo itself is already blurry           -> the problem is in the evolution optical flow
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nowcasting.models.nowcastnet import Net


def structure_score(field):
    """Structure sharpness: mean absolute gradient between adjacent pixels.
    Higher means crisper detail and edges."""
    gx = np.abs(np.diff(field, axis=-1)).mean()
    gy = np.abs(np.diff(field, axis=-2)).mean()
    return gx + gy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='model checkpoint (.ckpt)')
    p.add_argument('--input', required=True, help='input NetCDF (20 frames, dBZ x10 format)')
    p.add_argument('--out_dir', default='./diag_typhoon')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--warp_mode', default=None, choices=['bilinear', 'nearest'],
                   help='if omitted, use the setting stored in the checkpoint args (original: nearest)')
    args_cli = p.parse_args()
    os.makedirs(args_cli.out_dir, exist_ok=True)
    device = torch.device(args_cli.device)

    # -- Load checkpoint ------------------------------------------------------
    ckpt = torch.load(args_cli.ckpt, map_location='cpu', weights_only=False)
    ck_args = ckpt.get('args') if isinstance(ckpt, dict) else None
    sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    data_max = float(ckpt.get('data_max', 55.5)) if isinstance(ckpt, dict) else 55.5

    class A: pass
    a = A()
    for k, v in dict(input_length=20, total_length=50, img_height=512,
                     img_width=512, ngf=32).items():
        setattr(a, k, getattr(ck_args, k, v) if ck_args is not None else v)
    a.evo_ic = a.gen_oc = a.total_length - a.input_length
    a.ic_feature = a.ngf * 10
    # warp mode precedence: command line > checkpoint args > original 'nearest'
    a.warp_mode = args_cli.warp_mode or getattr(ck_args, 'warp_mode', 'nearest')

    print(f"[INFO] data_max={data_max:.2f} | warp_mode={a.warp_mode} | "
          f"input={a.input_length} pred={a.evo_ic}", flush=True)

    model = Net(a).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] Weights loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if len(missing) > 0:
        print("[WARNING] Some weights were not loaded; results may be unreliable!", flush=True)
    model.eval()

    # -- Read input NetCDF (same as training: /10 to dBZ, flip y, normalize) ---
    import xarray as xr
    ds = xr.open_dataset(args_cli.input, decode_times=False)
    var = list(ds.data_vars)[0]
    raw = ds[var].values.astype(np.float32)      # (T,H,W)
    ds.close()
    raw = raw / 10.0
    raw = raw[:, ::-1, :]
    print(f"[INFO] Input {var} shape={raw.shape} max={raw.max():.1f}dBZ", flush=True)

    T = a.total_length
    frames = np.zeros((1, T, a.img_height, a.img_width, 2), dtype=np.float32)
    n_in = min(a.input_length, raw.shape[0])
    frames[0, :n_in, :, :, 0] = raw[:n_in] / data_max
    frames[0, :n_in, :, :, 1] = 1.0
    x = torch.from_numpy(frames).to(device)

    # -- Single forward pass, extracting both evo (advection) and gen (final) --
    with torch.no_grad():
        gen_out, evo_result, motion = model(x, return_evo=True)
    gen = gen_out[0, ..., 0].cpu().numpy() * data_max     # (pred,H,W) dBZ
    evo = evo_result[0].cpu().numpy() * data_max
    mot = motion[0].cpu().numpy()                          # (pred,2,H,W)

    inp_last = raw[n_in - 1]
    s0 = structure_score(inp_last)

    print("\n" + "=" * 76, flush=True)
    print(f"Last input frame: structure={s0:.4f} max={inp_last.max():.1f}dBZ", flush=True)
    print(f"{'lead':>5} | {'EVO struct':>11} {'EVO kept%':>10} {'EVO max':>8} | "
          f"{'GEN struct':>11} {'GEN kept%':>10} {'GEN max':>8} | {'flow |v|':>9}", flush=True)
    for t in range(gen.shape[0]):
        se, sg = structure_score(evo[t]), structure_score(gen[t])
        vmag = np.sqrt(mot[t, 0] ** 2 + mot[t, 1] ** 2).mean()
        print(f"{t+1:>5} | {se:>9.4f} {100*se/s0:>8.1f}% {evo[t].max():>8.1f} | "
              f"{sg:>9.4f} {100*sg/s0:>8.1f}% {gen[t].max():>8.1f} | {vmag:>8.3f}", flush=True)

    # -- Figure: EVO vs GEN at several lead times -----------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        leads = [0, gen.shape[0] // 3, 2 * gen.shape[0] // 3, gen.shape[0] - 1]
        fig, axes = plt.subplots(2, len(leads), figsize=(4 * len(leads), 8))
        for c, t in enumerate(leads):
            axes[0, c].imshow(evo[t], vmin=0, vmax=60, cmap='jet')
            axes[0, c].set_title(f'EVO (advection) +{(t+1)*6}min'); axes[0, c].axis('off')
            axes[1, c].imshow(gen[t], vmin=0, vmax=60, cmap='jet')
            axes[1, c].set_title(f'GEN (final) +{(t+1)*6}min'); axes[1, c].axis('off')
        fig.suptitle('Top = evolution (pure advection)   Bottom = generation (final output), dBZ')
        fig.tight_layout()
        fp = os.path.join(args_cli.out_dir, 'evo_vs_gen.png')
        fig.savefig(fp, dpi=100); plt.close(fig)
        print(f"\n[INFO] Comparison figure: {fp}", flush=True)
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}", flush=True)

    np.save(os.path.join(args_cli.out_dir, 'evo.npy'), evo.astype(np.float16))
    np.save(os.path.join(args_cli.out_dir, 'gen.npy'), gen.astype(np.float16))
    print("\nInterpretation: EVO good but GEN blurry -> generative network / GAN;\n                 EVO already blurry -> evolution optical flow", flush=True)


if __name__ == '__main__':
    main()
