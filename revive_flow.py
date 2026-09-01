#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Revive the optical-flow output layer (outc_v) of the Evolution network.

Background (diagnosis):
  evo_net.outc_v.conv.weight had collapsed to std 1.2e-4 (the intensity branch is 0.186,
  a 1500x difference), degenerating the flow output into a "bias-only constant field":
    - flow magnitude |v| stuck at 0.03-0.23 px/frame (a real typhoon needs ~5 px/frame);
    - |v| stayed bit-for-bit identical no matter how lambda_motion / lambda_evo were tuned
      (a self-locking loop: output ~ constant -> gradient ~ 0 -> can never learn).
  This was the root cause of typhoons "staying put and then smearing out".

This script:
  Starting from an existing checkpoint, re-initializes only outc_v's weight/bias (all other
  weights are preserved) so the flow branch regains a useful gradient, then warm-start from it.

Usage:
    python revive_flow.py \
        --ckpt     ./checkpoints_v4/checkpoint_epoch_10.ckpt \
        --out      ./checkpoints_v4/revived_ep10.ckpt \
        --flow_std 0.05        # std of the re-initialized weights (default 0.05)

Then continue training with --pretrained_model pointing at the --out file.
"""
import os
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='input checkpoint')
    p.add_argument('--out', required=True, help='output (revived) checkpoint')
    p.add_argument('--flow_std', type=float, default=0.05,
                   help='std for re-initializing outc_v weights. Too small and it still cannot '
                        'learn; too large and early forecasts get noisy. Suggested 0.02-0.1')
    p.add_argument('--reset_bias', action='store_true', default=True,
                   help='also zero the bias (default on) -- the bias is the source of the '
                        'current constant flow field')
    p.add_argument('--also_reset_gamma', action='store_true',
                   help='also reset gamma to a small positive value (only if the intensity term also looks dead; off by default)')
    args = p.parse_args()

    print(f"[INFO] Reading: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise ValueError('no model_state_dict in checkpoint -- unexpected format')
    sd = ckpt['model_state_dict']

    # Locate the flow output layer (tolerates the DDP 'module.' prefix)
    wk = [k for k in sd if k.endswith('outc_v.conv.weight')]
    bk = [k for k in sd if k.endswith('outc_v.conv.bias')]
    if not wk:
        raise KeyError('outc_v.conv.weight not found -- is this checkpoint from this model?')
    wk, bk = wk[0], (bk[0] if bk else None)

    w = sd[wk]
    print(f"\n[Before revival] {wk}")
    print(f"    shape={tuple(w.shape)} std={w.float().std():.6e} absmax={w.float().abs().max():.6e}")
    if bk is not None:
        b = sd[bk]
        print(f"    bias: mean={b.float().mean():.6f} absmax={b.float().abs().max():.6f}")

    # -- Re-initialize with a normal distribution; std given on the command line --
    new_w = torch.randn_like(w.float()) * args.flow_std
    sd[wk] = new_w.to(w.dtype)
    if bk is not None and args.reset_bias:
        # Zero the bias: it is exactly the source of the current constant small displacement
        sd[bk] = torch.zeros_like(sd[bk])

    print(f"\n[After revival] {wk}")
    print(f"    std={sd[wk].float().std():.6e} absmax={sd[wk].float().abs().max():.6e}")
    if bk is not None and args.reset_bias:
        print(f"    bias zeroed (it was the source of the constant flow field)")

    if args.also_reset_gamma:
        gk = [k for k in sd if k.endswith('evo_net.gamma')]
        if gk:
            sd[gk[0]] = torch.full_like(sd[gk[0]], 0.01)
            print(f"    gamma reset to 0.01")

    # Optimizer state must be dropped: old momentum/second-moment terms belong to the dead
    # weights and would immediately drag the new ones back to zero
    for k in ('optimizer_state_dict', 'scheduler_state_dict'):
        if k in ckpt:
            ckpt[k] = None
    print("\n[INFO] Cleared optimizer/scheduler state (old momentum would pull new weights back to zero)")

    ckpt['model_state_dict'] = sd
    ckpt['revived_flow'] = {'flow_std': args.flow_std, 'src': os.path.abspath(args.ckpt)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[DONE] Saved: {args.out}")
    print("\nNext: warm-start training from it (the discriminator is still inherited automatically):")
    print(f"    --pretrained_model {args.out}")
    print("Also raise the learning rate a notch (e.g. --lr 1e-4) so the fresh flow layer can learn.")


if __name__ == '__main__':
    main()
