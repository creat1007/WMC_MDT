#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复活 evolution 网络的光流输出层（outc_v）。

背景（诊断结论）：
  evo_net.outc_v.conv.weight 的 std 已坍缩到 1.2e-4（强度分支是 0.186，差 1500 倍），
  光流输出退化成"只剩 bias 的常数场"，导致：
    - 流场 |v| 恒为 0.03~0.23 像素/帧（台风真实位移应为 ~5 像素/帧），
    - 无论怎么调 lambda_motion / lambda_evo，|v| 一模一样纹丝不动（自锁死循环：
      输出≈常数 → 梯度≈0 → 永远学不动）。
  → 台风"原地不动然后糊掉"的根因。

本脚本：
  从已有 checkpoint 出发，只重新初始化 outc_v 的 weight/bias（其余权重全部保留），
  让光流分支重新获得有效梯度，然后用它 warm start 继续训练。

用法：
    python revive_flow.py \
        --ckpt     ./checkpoints_v4/checkpoint_epoch_10.ckpt \
        --out      ./checkpoints_v4/revived_ep10.ckpt \
        --flow_std 0.05        # 重新初始化的权重标准差（默认 0.05）

之后用 --pretrained_model 指向 --out 产物继续训练即可。
"""
import os
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='输入 checkpoint')
    p.add_argument('--out', required=True, help='输出（复活后的）checkpoint')
    p.add_argument('--flow_std', type=float, default=0.05,
                   help='outc_v 权重重新初始化的标准差。太小仍会学不动，太大初期预报会乱；'
                        '建议 0.02~0.1，默认 0.05')
    p.add_argument('--reset_bias', action='store_true', default=True,
                   help='同时把 bias 清零（默认开）。bias 是当前"常数流场"的来源')
    p.add_argument('--also_reset_gamma', action='store_true',
                   help='顺带把 gamma 重置为小正值（若强度项也疑似失效再用，默认不动）')
    args = p.parse_args()

    print(f"[INFO] 读取: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise ValueError('checkpoint 里没有 model_state_dict，格式不对')
    sd = ckpt['model_state_dict']

    # 定位光流输出层（兼容 DDP 的 module. 前缀）
    wk = [k for k in sd if k.endswith('outc_v.conv.weight')]
    bk = [k for k in sd if k.endswith('outc_v.conv.bias')]
    if not wk:
        raise KeyError('没找到 outc_v.conv.weight，检查 checkpoint 是否为本模型')
    wk, bk = wk[0], (bk[0] if bk else None)

    w = sd[wk]
    print(f"\n[复活前] {wk}")
    print(f"    shape={tuple(w.shape)} std={w.float().std():.6e} absmax={w.float().abs().max():.6e}")
    if bk is not None:
        b = sd[bk]
        print(f"    bias: mean={b.float().mean():.6f} absmax={b.float().abs().max():.6f}")

    # ── 重新初始化：正态分布，std 由命令行给定 ──────────────────────────
    new_w = torch.randn_like(w.float()) * args.flow_std
    sd[wk] = new_w.to(w.dtype)
    if bk is not None and args.reset_bias:
        # bias 清零：它正是当前"恒定小位移"的来源，必须清掉
        sd[bk] = torch.zeros_like(sd[bk])

    print(f"\n[复活后] {wk}")
    print(f"    std={sd[wk].float().std():.6e} absmax={sd[wk].float().abs().max():.6e}")
    if bk is not None and args.reset_bias:
        print(f"    bias 已清零（原本是恒定流场的来源）")

    if args.also_reset_gamma:
        gk = [k for k in sd if k.endswith('evo_net.gamma')]
        if gk:
            sd[gk[0]] = torch.full_like(sd[gk[0]], 0.01)
            print(f"    gamma 已重置为 0.01")

    # 优化器状态必须丢弃：旧的动量/二阶矩对应的是已死的权重，会立刻把新权重拉回零
    for k in ('optimizer_state_dict', 'scheduler_state_dict'):
        if k in ckpt:
            ckpt[k] = None
    print("\n[INFO] 已清空 optimizer/scheduler 状态（旧动量会把新权重拽回零）")

    ckpt['model_state_dict'] = sd
    ckpt['revived_flow'] = {'flow_std': args.flow_std, 'src': os.path.abspath(args.ckpt)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[DONE] 已保存: {args.out}")
    print("\n下一步：用它 warm start 继续训练（判别器仍会自动继承）：")
    print(f"    --pretrained_model {args.out}")
    print("建议同时把学习率抬高一档（如 --lr 1e-4），让新初始化的光流层学得动。")


if __name__ == '__main__':
    main()
