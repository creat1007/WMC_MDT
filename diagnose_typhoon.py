#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
台风"散架"诊断：判断结构是在 evolution(平流) 阶段丢的，还是 generation(生成) 阶段丢的。

模型分两段：
  ① evolution 网络 → evo_result（纯平流+强度演变，无 GAN 参与）
  ② generative 网络 → gen_result（最终输出，GAN 精修过）
本脚本一次推理同时取出两者，逐 lead-time 输出结构指标并出图对比。

用法（在有 checkpoint 和输入 nc 的机器上）：
    python diagnose_typhoon.py \
        --ckpt   /path/to/checkpoint_epoch_25.ckpt \
        --input  /path/to/台风个例的 input.nc \
        --out_dir ./diag_typhoon

看结果：
  - 若 evo 结构保持得好、gen 糊掉  → 问题在【生成网络/GAN】
  - 若 evo 本身就糊               → 问题在【evolution 光流】
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nowcasting.models.nowcastnet import Net


def structure_score(field):
    """结构锐利度：相邻像素梯度的平均绝对值。越高说明细节/边界越清晰。"""
    gx = np.abs(np.diff(field, axis=-1)).mean()
    gy = np.abs(np.diff(field, axis=-2)).mean()
    return gx + gy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='模型 checkpoint (.ckpt)')
    p.add_argument('--input', required=True, help='输入 nc（20帧，dBZ×10 格式）')
    p.add_argument('--out_dir', default='./diag_typhoon')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--warp_mode', default=None, choices=['bilinear', 'nearest'],
                   help='不指定则用 checkpoint 里 args 的设置（默认原版 nearest）')
    args_cli = p.parse_args()
    os.makedirs(args_cli.out_dir, exist_ok=True)
    device = torch.device(args_cli.device)

    # ── 载入 checkpoint ───────────────────────────────────────────────
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
    # warp 模式：命令行 > checkpoint 里的 args > 原版 nearest
    a.warp_mode = args_cli.warp_mode or getattr(ck_args, 'warp_mode', 'nearest')

    print(f"[INFO] data_max={data_max:.2f} | warp_mode={a.warp_mode} | "
          f"input={a.input_length} pred={a.evo_ic}", flush=True)

    model = Net(a).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] 权重加载: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if len(missing) > 0:
        print("[WARNING] 有权重未加载，结果可能不可信！", flush=True)
    model.eval()

    # ── 读输入 nc（与训练一致：/10 转 dBZ、翻转 y、归一化）─────────────
    import xarray as xr
    ds = xr.open_dataset(args_cli.input, decode_times=False)
    var = list(ds.data_vars)[0]
    raw = ds[var].values.astype(np.float32)      # (T,H,W)
    ds.close()
    raw = raw / 10.0
    raw = raw[:, ::-1, :]
    print(f"[INFO] 输入 {var} shape={raw.shape} max={raw.max():.1f}dBZ", flush=True)

    T = a.total_length
    frames = np.zeros((1, T, a.img_height, a.img_width, 2), dtype=np.float32)
    n_in = min(a.input_length, raw.shape[0])
    frames[0, :n_in, :, :, 0] = raw[:n_in] / data_max
    frames[0, :n_in, :, :, 1] = 1.0
    x = torch.from_numpy(frames).to(device)

    # ── 一次前向，同时取 evo(平流) 和 gen(最终) ──────────────────────
    with torch.no_grad():
        gen_out, evo_result, motion = model(x, return_evo=True)
    gen = gen_out[0, ..., 0].cpu().numpy() * data_max     # (pred,H,W) dBZ
    evo = evo_result[0].cpu().numpy() * data_max
    mot = motion[0].cpu().numpy()                          # (pred,2,H,W)

    inp_last = raw[n_in - 1]
    s0 = structure_score(inp_last)

    print("\n" + "=" * 76, flush=True)
    print(f"输入末帧: 结构={s0:.4f} max={inp_last.max():.1f}dBZ", flush=True)
    print(f"{'lead':>5} | {'EVO结构':>9} {'EVO保留%':>9} {'EVO max':>8} | "
          f"{'GEN结构':>9} {'GEN保留%':>9} {'GEN max':>8} | {'流场|v|':>8}", flush=True)
    for t in range(gen.shape[0]):
        se, sg = structure_score(evo[t]), structure_score(gen[t])
        vmag = np.sqrt(mot[t, 0] ** 2 + mot[t, 1] ** 2).mean()
        print(f"{t+1:>5} | {se:>9.4f} {100*se/s0:>8.1f}% {evo[t].max():>8.1f} | "
              f"{sg:>9.4f} {100*sg/s0:>8.1f}% {gen[t].max():>8.1f} | {vmag:>8.3f}", flush=True)

    # ── 出图：输入末帧 / EVO / GEN 在几个 lead time 的对比 ────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        leads = [0, gen.shape[0] // 3, 2 * gen.shape[0] // 3, gen.shape[0] - 1]
        fig, axes = plt.subplots(2, len(leads), figsize=(4 * len(leads), 8))
        for c, t in enumerate(leads):
            axes[0, c].imshow(evo[t], vmin=0, vmax=60, cmap='jet')
            axes[0, c].set_title(f'EVO(平流) +{(t+1)*6}min'); axes[0, c].axis('off')
            axes[1, c].imshow(gen[t], vmin=0, vmax=60, cmap='jet')
            axes[1, c].set_title(f'GEN(最终) +{(t+1)*6}min'); axes[1, c].axis('off')
        fig.suptitle('上=evolution纯平流  下=generation最终输出 (dBZ)')
        fig.tight_layout()
        fp = os.path.join(args_cli.out_dir, 'evo_vs_gen.png')
        fig.savefig(fp, dpi=100); plt.close(fig)
        print(f"\n[INFO] 对比图: {fp}", flush=True)
    except Exception as e:
        print(f"[WARN] 出图失败: {e}", flush=True)

    np.save(os.path.join(args_cli.out_dir, 'evo.npy'), evo.astype(np.float16))
    np.save(os.path.join(args_cli.out_dir, 'gen.npy'), gen.astype(np.float16))
    print("\n判读：EVO保持好而GEN糊 → 问题在生成网络/GAN；EVO本身就糊 → 问题在evolution光流", flush=True)


if __name__ == '__main__':
    main()
