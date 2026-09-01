import os
import sys
import time
import argparse
import traceback
import gc
import signal
import csv
import json
import math
import queue as queue_module
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nowcasting.models.nowcastnet import Net
from nowcasting.data_provider.train_loader import RadarTrainDataset, compute_data_stats
from nowcasting.layers.generation.discriminator import (
    Temporal_Discriminator, hinge_loss_d, hinge_loss_g, pool_regularization,
)

# Mixed multi-region training: comma-separated directories (used when --train_data_path is omitted)
TRAIN_DATA_PATH = '/path/to/radar_data/region_A,/path/to/radar_data/region_B'
CACHE_DIR = '/path/to/radar_data/cache_512/'
SAVE_DIR = './checkpoints'
LOG_FILE = './checkpoints/training_log.csv'

class ParallelBatchLoader:
    """
    Thread-based parallel data loader that sidesteps the /dev/shm limit entirely.
    Multiprocess DataLoader workers ship tensors through shared memory (Docker defaults to
    64MB, which is not enough). With threads, all workers share the main process memory and
    incur no shm overhead. xarray releases the GIL while reading NetCDF, so threads give
    genuine I/O parallelism.
    """
    def __init__(self, dataset, sampler, batch_size, num_threads=8, prefetch=2):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.prefetch = prefetch
        # Read samples in parallel only when the .npy cache is used (mmap, thread-safe).
        # Falling back to raw netCDF requires sequential reads (HDF5/netCDF4 is not
        # thread-safe and concurrent access segfaults).
        self.parallel = getattr(dataset, 'cache_dir', None) is not None and num_threads > 1
        self._pool = ThreadPoolExecutor(max_workers=num_threads) if self.parallel else None

    def _make_batches(self):
        indices = list(self.sampler)
        return [
            indices[i: i + self.batch_size]
            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size)
        ]

    def _load_batch(self, batch_indices):
        if self.parallel:
            # Cached mode: read in parallel (mmap reads + numpy copies release the GIL),
            # keeping the GPU fed
            samples = list(self._pool.map(self.dataset.__getitem__, batch_indices))
        else:
            samples = [self.dataset.__getitem__(idx) for idx in batch_indices]
        return torch.stack(samples)

    def __iter__(self):
        batches = self._make_batches()
        q = queue_module.Queue(maxsize=self.prefetch)

        def producer():
            for batch_indices in batches:
                q.put(self._load_batch(batch_indices))
            q.put(None)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            batch = q.get()
            if batch is None:
                break
            yield batch

    def __len__(self):
        return len(list(self.sampler)) // self.batch_size


class GracefulExiter:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        signal.signal(signal.SIGINT, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        print(f"\n[INFO] Received signal {signum}; saving the model and exiting...", flush=True)
        self.should_exit = True

class MemoryMonitor:
    def __init__(self, device, local_rank=0):
        self.device = device
        self.local_rank = local_rank
        self.peak_gpu_mem = 0
        self.peak_cpu_mem = 0
        
    def get_gpu_mem(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return allocated, reserved
        return 0, 0
    
    def get_cpu_mem(self):
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        mem_kb = int(line.split()[1])
                        return mem_kb / 1024 / 1024
        except Exception as e:
            pass
        return 0
    
    def get_cpu_mem_percent(self):
        try:
            with open('/proc/meminfo', 'r') as f:
                total = 0
                for line in f:
                    if line.startswith('MemTotal:'):
                        total = int(line.split()[1])
                        break
            if total > 0:
                current = self.get_cpu_mem() * 1024 * 1024
                return (current / total) * 100
        except:
            pass
        return 0
    
    def update_peak(self):
        gpu_alloc, _ = self.get_gpu_mem()
        cpu_mem = self.get_cpu_mem()
        self.peak_gpu_mem = max(self.peak_gpu_mem, gpu_alloc)
        self.peak_cpu_mem = max(self.peak_cpu_mem, cpu_mem)
    
    def print_status(self, step_name=""):
        if self.local_rank == 0:
            gpu_alloc, gpu_reserved = self.get_gpu_mem()
            cpu_mem = self.get_cpu_mem()
            cpu_percent = self.get_cpu_mem_percent()
            print(f"[MEM] {step_name} | CPU: {cpu_mem:.1f}GB ({cpu_percent:.1f}%) | GPU: {gpu_alloc:.1f}/{gpu_reserved:.1f}GB", flush=True)
    
    def cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

class SobelGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx, persistent=False)
        self.register_buffer('ky', ky, persistent=False)

    def sobel_mag(self, x):
        gx = F.conv2d(x, self.kx.to(x.dtype), padding=1)
        gy = F.conv2d(x, self.ky.to(x.dtype), padding=1)
        return (gx.pow(2) + gy.pow(2) + 1e-8).sqrt()

    def forward(self, pred, target):
        b, t, h, w = pred.shape
        pred_2d = pred.reshape(b * t, 1, h, w)
        target_2d = target.reshape(b * t, 1, h, w)
        return F.l1_loss(self.sobel_mag(pred_2d), self.sobel_mag(target_2d))

def radar_weight(target_norm, data_max):
    """
    Graded foreground weighting in the DGMR / NowcastNet style, based on true dBZ
    thresholds. Over 99% of radar pixels are zero or weak echo, so strong echoes need a
    much higher weight -- otherwise the model collapses to the trivial "blank output"
    solution, which minimizes the loss but has no forecast value.
    """
    dbz = target_norm * data_max
    w = torch.ones_like(dbz)
    w = torch.where(dbz >= 10.0, torch.full_like(w, 2.0), w)
    w = torch.where(dbz >= 20.0, torch.full_like(w, 3.0), w)
    w = torch.where(dbz >= 30.0, torch.full_like(w, 5.0), w)
    w = torch.where(dbz >= 40.0, torch.full_like(w, 10.0), w)  # cap lowered 30 -> 10 to tame gradient spikes at small batch sizes
    return w

def balanced_l1(pred, target, mask, data_max, fg_thresh_dbz=15.0, bg_weight=0.2):
    """Frequency-balanced L1: average foreground and background separately, then combine.

    Key point: never take a weighted average over *all* pixels. Radar fields are ~99.99%
    zero background, so any sum/total normalization dilutes foreground error down to ~1e-5
    and the model collapses to blank output. Here the foreground is averaged over its own
    pixel count, so every strong-echo pixel receives a strong gradient; the background
    contributes with a small weight (bg_weight) to avoid full-screen noise. Within the
    foreground, dBZ-graded weighting (radar_weight) penalizes strong echoes more."""
    diff = (pred - target).abs() * mask
    w = radar_weight(target, data_max)
    dbz = target * data_max
    fg = (dbz >= fg_thresh_dbz).float() * mask
    bg = (dbz < fg_thresh_dbz).float() * mask

    fg_loss = (w * diff * fg).sum() / (fg.sum() + 1e-6)
    bg_loss = (diff * bg).sum() / (bg.sum() + 1e-6)
    return fg_loss + bg_weight * bg_loss

def motion_smoothness(motion):
    """Spatial smoothness regularization for the flow field: motion (B, T, 2, H, W).
    Encourages the evolution network to learn a continuous, physically plausible advection
    field rather than scattered displacements."""
    b, t, c, h, w = motion.shape
    m = motion.reshape(b * t, c, h, w)
    dx = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
    dy = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
    return dx + dy


@torch.no_grad()
def evaluate_csi(raw_model, val_dataset, device, args, max_samples=200):
    """Compute CSI (Critical Success Index) on the held-out validation set -- the standard
    nowcasting score. A blank forecast scores CSI=0, so this immediately exposes the
    "loss is falling but the model is useless" illusion.
    Returns (dict of per-threshold CSI, mean CSI). Call on rank 0 only."""
    raw_model.eval()
    thresholds = [20.0, 30.0, 40.0]
    agg = {t: [0, 0, 0] for t in thresholds}  # [hits, misses, false_alarms]
    n = min(max_samples, len(val_dataset))
    if n == 0:
        raw_model.train()
        return {t: 0.0 for t in thresholds}, 0.0
    idxs = np.linspace(0, len(val_dataset) - 1, n).astype(int)
    for i in idxs:
        vid = val_dataset[int(i)].unsqueeze(0).to(device)
        with autocast('cuda', enabled=True):
            pred = raw_model(vid)[..., 0]
        target = vid[:, args.input_length:, :, :, 0]
        mask = vid[:, args.input_length:, :, :, 1] > 0
        pred_dbz = pred.float() * args.data_max
        target_dbz = target.float() * args.data_max
        for t in thresholds:
            p = (pred_dbz >= t) & mask
            g = (target_dbz >= t) & mask
            agg[t][0] += int((p & g).sum().item())
            agg[t][1] += int((~p & g).sum().item())
            agg[t][2] += int((p & ~g).sum().item())
    csi = {}
    for t in thresholds:
        h, m, f = agg[t]
        csi[t] = h / (h + m + f + 1e-6)
    raw_model.train()
    mean_csi = sum(csi.values()) / len(csi)
    return csi, mean_csi

def train_one_epoch(model, loader, optimizer, scaler, device, args, grad_loss_fn, local_rank=0,
                    disc=None, optimizer_d=None, scaler_d=None, world_size=1, adv_active=True):
    """Train one epoch.
    - disc=None: plain L1 training (original behavior, unchanged).
    - disc!=None: alternating adversarial (GAN) training. disc is a *raw* module (not wrapped
      in DDP); under multi-GPU its gradients are all-reduced manually to stay in sync, while
      the generator `model` remains DDP-wrapped.
    - adv_active=False: discriminator warm-up -- D still trains, but the adversarial loss is
      not yet added to G.
    """
    model.train()
    if disc is not None:
        disc.train()
    running_loss = 0.0
    running_d = 0.0
    valid_batches = 0
    nan_batches = 0

    for batch_idx, frames in enumerate(loader):
        if hasattr(train_one_epoch, 'exiter') and train_one_epoch.exiter.should_exit:
            if local_rank == 0:
                print("[INFO] Exit signal received; stopping training...", flush=True)
            break

        try:
            frames = frames.to(device, non_blocking=True)
            target = frames[:, args.input_length:, :, :, 0]
            mask = frames[:, args.input_length:, :, :, 1]

            # -- Generator forward + base L1 / structure / evolution losses ------
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=True):
                gen_out, evo_result, motion = model(frames, return_evo=True)
                pred = gen_out[..., 0]
                gen_l1 = balanced_l1(pred, target, mask, args.data_max,
                                     args.fg_thresh_dbz, args.bg_weight)
                grad = grad_loss_fn(pred * mask, target * mask)
                evo_l1 = balanced_l1(evo_result, target, mask, args.data_max,
                                     args.fg_thresh_dbz, args.bg_weight)
                motion_reg = motion_smoothness(motion)
                g_loss = (gen_l1
                          + args.lambda_grad * grad
                          + args.lambda_evo * evo_l1
                          + args.lambda_motion * motion_reg)

            # -- GAN: one discriminator step + the generator's adversarial term ---
            # The discriminator runs entirely in float32 without autocast/scaler_d, which
            # completely avoids the AMP+GAN "unscale FP16 gradients" conflict. It is small
            # (1.2M params), so the float32 cost is negligible.
            if disc is not None:
                input_seq = frames[:, :args.input_length, :, :, 0].float()   # (B, IL, H, W)
                real_seq = frames[:, :, :, :, 0].float()                     # (B, total, H, W)
                fake_seq = torch.cat([input_seq, pred.float()], dim=1)       # (B, total, H, W)

                # (1) Train the discriminator (float32, plain backward, no scaler)
                optimizer_d.zero_grad(set_to_none=True)
                d_real = disc(real_seq)
                d_fake = disc(fake_seq.detach())
                d_loss = hinge_loss_d(d_real, d_fake)
                d_loss.backward()
                if dist.is_initialized():                                    # sync D grads manually
                    for p in disc.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(world_size)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                optimizer_d.step()
                dv = d_loss.item()
                running_d += dv if math.isfinite(dv) else 0.0

                # (2) Generator adversarial term: freeze D's parameters (gradients still flow
                #     back to G) and add it to g_loss. D stays frozen until G's backward is
                #     done (it is unfrozen after the G step below).
                if adv_active:
                    for p in disc.parameters():
                        p.requires_grad_(False)
                    g_adv = hinge_loss_g(disc(fake_seq))                      # float32
                    pool = pool_regularization(pred.float(), target.float())
                    g_loss = g_loss + args.lambda_adv * g_adv + args.lambda_pool * pool

            # -- Generator step (every rank must backward, or DDP deadlocks) ------
            scaler.scale(g_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            # Unfreeze D after the G step so the next batch can train the discriminator
            if disc is not None and adv_active:
                for p in disc.parameters():
                    p.requires_grad_(True)

            loss_val = g_loss.item()
            if math.isfinite(loss_val):
                running_loss += loss_val
                valid_batches += 1
            else:
                nan_batches += 1
                if local_rank == 0 and nan_batches <= 5:
                    print(f"[WARNING] NaN/Inf loss at batch {batch_idx} "
                          f"(loss={loss_val}); GradScaler skipped this update automatically", flush=True)

        except RuntimeError as e:
            if "out of memory" in str(e):
                if local_rank == 0:
                    print(f"[ERROR] GPU OOM at batch {batch_idx}, skipping...", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

    if local_rank == 0 and nan_batches > 0:
        print(f"[WARNING] Skipped {nan_batches} NaN/Inf batches this epoch", flush=True)
    if local_rank == 0 and disc is not None and valid_batches > 0:
        print(f"[GAN] Mean discriminator loss this epoch: {running_d / max(valid_batches,1):.4f} "
              f"| adversarial term active: {adv_active}", flush=True)

    if valid_batches == 0:
        avg_loss = float('nan')
    else:
        avg_loss = running_loss / valid_batches

    if dist.is_initialized():
        loss_tensor = torch.tensor(avg_loss if math.isfinite(avg_loss) else 0.0).to(device)
        count_tensor = torch.tensor(float(valid_batches)).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = (loss_tensor / count_tensor).item() if count_tensor.item() > 0 else float('nan')
    else:
        global_avg_loss = avg_loss

    return global_avg_loss

def init_csv_log(log_file, local_rank=0):
    if local_rank != 0:
        return
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    file_exists = os.path.isfile(log_file)
    if not file_exists:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'lr', 'elapsed_s',
                             'val_csi_mean', 'csi20', 'csi30', 'csi40'])

def append_csv_log(log_file, epoch, train_loss, lr, elapsed_s,
                   csi_mean, csi20, csi30, csi40, local_rank=0):
    if local_rank != 0:
        return
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f'{train_loss:.6f}', lr, f'{elapsed_s:.1f}',
                         f'{csi_mean:.4f}', f'{csi20:.4f}', f'{csi30:.4f}', f'{csi40:.4f}'])

def setup_distributed():
    # Raise the NCCL heartbeat timeout before init (default is 480s) so that ranks 1..N
    # waiting at the barrier do not trip the watchdog while rank 0 scans the dataset
    os.environ.setdefault('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '1800')

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(hours=2),  # dataset scanning can exceed the 10-minute default
        )
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def unwrap(model):
    """Return the raw model without the DDP wrapper (on a single GPU, model is already raw)."""
    return model.module if hasattr(model, 'module') else model

def main():
    rank, world_size, local_rank = setup_distributed()
    
    parser = argparse.ArgumentParser(description='NowcastNet Multi-GPU Training')
    parser.add_argument('--train_data_path', type=str, default=TRAIN_DATA_PATH)
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR,
                        help='Preprocessed .npy cache dir (produced by preprocess_cache.py). '
                             'When set, data is memory-mapped directly for much faster loading. '
                             'Use "none" to disable.')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--log_file', type=str, default=LOG_FILE)
    
    parser.add_argument('--batch_size', type=int, default=16, help='per-GPU batch size')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=12, help='data-loading threads per GPU')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    
    parser.add_argument('--cleanup_interval', type=int, default=20)
    parser.add_argument('--monitor_interval', type=int, default=100)
    
    parser.add_argument('--input_length', type=int, default=20)
    parser.add_argument('--total_length', type=int, default=50)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--warp_mode', type=str, default='bilinear', choices=['bilinear', 'nearest'],
                        help='Interpolation mode for the iterative warp in evolution. '
                             '"bilinear" enables sub-pixel motion so fine structure survives the '
                             '30-step rollout; "nearest" is the original behavior.')
    parser.add_argument('--lambda_grad', type=float, default=0.1)
    parser.add_argument('--lambda_evo', type=float, default=1.0,
                        help='weight of the evolution self-supervision loss')
    parser.add_argument('--lambda_motion', type=float, default=0.01,
                        help='weight of the flow-field smoothness regularization')
    parser.add_argument('--fg_thresh_dbz', type=float, default=15.0,
                        help='foreground/background threshold (dBZ); pixels at or above it are averaged as foreground')
    parser.add_argument('--bg_weight', type=float, default=0.2,
                        help='background loss weight -- suppresses full-screen noise without letting background dominate')
    # -- GAN / discriminator (adversarial training; fixes blurring and intensity collapse) --
    parser.add_argument('--gan', action='store_true',
                        help='enable adversarial training. Recommended to warm start from a best_model via --pretrained_model')
    parser.add_argument('--lambda_adv', type=float, default=0.01,
                        help='adversarial loss weight for the generator. Too high destabilizes training, too low has no effect; start at 0.01')
    parser.add_argument('--lambda_pool', type=float, default=1.0,
                        help='pooling regularization weight; constrains coarse-scale rainfall so the discriminator cannot force fabricated strong echoes')
    parser.add_argument('--disc_lr', type=float, default=2e-4,
                        help='discriminator learning rate')
    parser.add_argument('--disc_base_c', type=int, default=32,
                        help='discriminator base channel width')
    parser.add_argument('--disc_warmup_epochs', type=int, default=1,
                        help='discriminator warm-up epochs: train D only, without adding the adversarial term to G')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='fraction of files held out for validation (temporally independent)')
    parser.add_argument('--val_interval', type=int, default=2,
                        help='compute validation CSI every N epochs')
    parser.add_argument('--val_max_samples', type=int, default=200,
                        help='maximum samples evaluated per validation pass (bounds runtime)')
    parser.add_argument('--use_compile', action='store_true', default=False)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='load only model weights as a starting point (fine-tune); optimizer/epoch are not restored')
    parser.add_argument('--data_max', dest='data_max_fixed', type=float, default=0.0,
                        help='Pin the normalization reference data_max (>0 to take effect). When '
                             'warm-starting on a different dataset it MUST match the pretrained '
                             'value; if left unset it is inherited from --pretrained_model or '
                             'computed automatically.')
    
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    
    args = parser.parse_args()

    # Important: when num_workers=0, prefetch_factor must be None
    if args.num_workers == 0:
        args.prefetch_factor = None

    # Allow --cache_dir none to explicitly disable the cache (fall back to reading NetCDF)
    if args.cache_dir is not None and args.cache_dir.strip().lower() in ('none', ''):
        args.cache_dir = None

    global_batch_size = args.batch_size * world_size
    if rank == 0:
        print(f"[INFO] Distributed training on {world_size} GPU(s)", flush=True)
        print(f"[INFO] Per-GPU batch size: {args.batch_size}", flush=True)
        print(f"[INFO] Global batch size: {global_batch_size}", flush=True)
        print(f"[INFO] Learning rate: {args.lr}", flush=True)

    args.evo_ic = args.total_length - args.input_length
    args.gen_oc = args.total_length - args.input_length
    args.ic_feature = args.ngf * 10

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        # Every process must set these, not just rank 0
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        if rank == 0:
            print(f"[INFO] GPU: {torch.cuda.get_device_name(local_rank)}", flush=True)
            print(f"[INFO] Per-GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f}GB", flush=True)
            print(f"[INFO] Total memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3 * world_size:.1f}GB", flush=True)

    mem_monitor = MemoryMonitor(device, local_rank)
    exiter = GracefulExiter()
    train_one_epoch.exiter = exiter

    if rank == 0:
        print("="*60, flush=True)
        print(f"[INFO] Starting distributed training", flush=True)
        print(f"[INFO] Per-GPU batch size: {args.batch_size}", flush=True)
        print(f"[INFO] Gradient accumulation steps: {args.grad_accum_steps}", flush=True)
        print(f"[INFO] Initial learning rate: {args.lr}", flush=True)
        print(f"[INFO] LR decay factor: {args.lr_factor}", flush=True)
        print(f"[INFO] LR patience: {args.lr_patience}", flush=True)
        print(f"[INFO] Minimum learning rate: {args.lr_min}", flush=True)
        print(f"[INFO] Workers per GPU: {args.num_workers}", flush=True)
        print(f"[INFO] Log file: {args.log_file}", flush=True)
        print("="*60, flush=True)

    init_csv_log(args.log_file, rank)

    try:
        # -- Step 1: determine the normalization reference data_max ------------
        # Precedence: --data_max > the value stored in --pretrained_model > auto-computed.
        # A warm start (especially onto a different dataset) MUST reuse the pretrained
        # data_max; changing the normalization scale invalidates all learned weights.
        cache_file = os.path.join(args.save_dir, 'data_stats.json')

        pinned, src = None, ''
        if getattr(args, 'data_max_fixed', 0) and args.data_max_fixed > 0:
            pinned, src = float(args.data_max_fixed), '--data_max flag'
        elif args.pretrained_model and os.path.exists(args.pretrained_model):
            try:
                _c = torch.load(args.pretrained_model, map_location='cpu', weights_only=False)
                if isinstance(_c, dict) and _c.get('data_max'):
                    pinned, src = float(_c['data_max']), 'inherited from pretrained_model'
                del _c
            except Exception:
                pinned = None

        if pinned is not None:
            data_max = pinned
            if rank == 0:
                print(f"[INFO] Using pinned data_max={data_max:.4f} ({src}; skipping the statistics scan)", flush=True)
        else:
            if rank == 0:
                data_max = compute_data_stats(args.train_data_path, percentile=99.9, cache_file=cache_file)
            # Barrier so the other ranks read the cache only after rank 0 has written it
            if world_size > 1:
                dist.barrier()
            if rank != 0:
                with open(cache_file, 'r') as f:
                    data_max = float(json.load(f)['data_max'])

        args.data_max = data_max
        # Strong-echo weighting threshold: ~32 units in the original space, scaled by data_max
        args.heavy_rain_threshold = 32.0 / args.data_max

        if rank == 0:
            print(f"[INFO] data_max={args.data_max:.4f}, strong-echo threshold (normalized)={args.heavy_rain_threshold:.4f}", flush=True)

        # -- Step 2: load the dataset (only rank 0 scans disk, then broadcasts) --
        if world_size > 1:
            dist.barrier()

        import pickle
        samples_cache_file = os.path.join(args.save_dir, 'samples_cache.pkl')

        def _build_dataset(samples):
            return RadarTrainDataset(
                data_path=args.train_data_path,
                input_length=args.input_length,
                total_length=args.total_length,
                img_height=args.img_height,
                img_width=args.img_width,
                data_max=args.data_max,
                _prebuilt_samples=samples,
                cache_dir=args.cache_dir,
            )

        start_time = time.time()
        if rank == 0:
            print("[INFO] Loading dataset (rank 0 scanning)...", flush=True)
            if os.path.exists(samples_cache_file):
                with open(samples_cache_file, 'rb') as f:
                    all_samples = pickle.load(f)
                print(f"[INFO] Loaded sample index from cache: {len(all_samples)} samples", flush=True)
            else:
                full = RadarTrainDataset(
                    data_path=args.train_data_path,
                    input_length=args.input_length,
                    total_length=args.total_length,
                    img_height=args.img_height,
                    img_width=args.img_width,
                    data_max=args.data_max,
                )
                all_samples = full.samples
                with open(samples_cache_file, 'wb') as f:
                    pickle.dump(all_samples, f)
                print(f"[INFO] Sample index cached to: {samples_cache_file}", flush=True)

            # -- Hold out validation by file (temporally independent, no train/val leakage) --
            files = sorted(set(s[0] for s in all_samples))
            n_val_files = max(1, int(round(len(files) * args.val_fraction)))
            val_files = set(files[-n_val_files:])
            train_samples = [s for s in all_samples if s[0] not in val_files]
            val_samples = [s for s in all_samples if s[0] in val_files]
            load_time = time.time() - start_time
            print(f"[INFO] Dataset ready in {load_time:.1f}s", flush=True)
            print(f"[INFO] Train: {len(train_samples)} samples from {len(files)-n_val_files} files; "
                  f"val: {len(val_samples)} samples from {n_val_files} files", flush=True)
            samples_to_broadcast = [train_samples, val_samples]
        else:
            samples_to_broadcast = [None, None]

        # Broadcast the training index to the other ranks (validation runs on rank 0 only)
        if world_size > 1:
            dist.broadcast_object_list(samples_to_broadcast, src=0)

        dataset = _build_dataset(samples_to_broadcast[0])

        # The validation set is built on rank 0 only
        val_dataset = _build_dataset(samples_to_broadcast[1]) if rank == 0 else None

        # Synchronize all processes
        if world_size > 1:
            dist.barrier()

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        
        if args.num_workers > 0:
            # Thread-based loader: bypasses the Docker /dev/shm limit; parallel I/O needs no shared memory
            train_loader = ParallelBatchLoader(
                dataset,
                sampler=sampler,
                batch_size=args.batch_size,
                num_threads=args.num_workers,
                prefetch=args.prefetch_factor,
            )
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
        if rank == 0:
            print(f"[INFO] DataLoader ready: ~{len(train_loader)} batches per GPU", flush=True)

        if rank == 0:
            print("[INFO] Initializing model...", flush=True)
        model = Net(args).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if rank == 0:
            print(f"[INFO] Model parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable", flush=True)
        
        # Load pretrained weights as a fine-tuning start (optimizer/epoch are not restored)
        pretrained_disc_sd = None   # if the ckpt carries a discriminator, inherit it later
        if args.pretrained_model and os.path.exists(args.pretrained_model):
            if rank == 0:
                print(f"[INFO] Loading pretrained weights: {args.pretrained_model}", flush=True)
            ckpt = torch.load(args.pretrained_model, map_location=device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if isinstance(ckpt, dict):
                pretrained_disc_sd = ckpt.get('disc_state_dict', None)
            if rank == 0:
                print(f"[INFO] Pretrained weights loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

        if args.use_compile and hasattr(torch, 'compile'):
            if rank == 0:
                print("[INFO] Applying torch.compile...", flush=True)
            model = torch.compile(model)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        elif rank == 0:
            print("[INFO] Single-GPU training; skipping the DDP wrapper", flush=True)

        scaled_lr = args.lr * math.sqrt(world_size)
        if rank == 0:
            print(f"[INFO] Scaled learning rate: {scaled_lr:.2e} ({args.lr:.2e} * sqrt({world_size}))", flush=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=1e-5, fused=False)
        
        # Schedule on validation CSI (higher is better). A blank forecast cannot sneak past
        # a CSI of 0, making this far more reliable than watching train_loss (with loss ~1e-5
        # the old threshold never triggered and the LR never decayed).
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )
        
        scaler = GradScaler('cuda')
        grad_loss_fn = SobelGradientLoss().to(device)

        # -- GAN discriminator: a raw module (no DDP); gradients are synced manually inside
        #    train_one_epoch under multi-GPU --
        disc = optimizer_d = scaler_d = None
        if args.gan:
            disc = Temporal_Discriminator(args.total_length, base_c=args.disc_base_c).to(device)
            # If the pretrained ckpt carries a discriminator (from a previous GAN run),
            # inherit it so D does not restart from scratch
            if pretrained_disc_sd is not None:
                try:
                    disc.load_state_dict(pretrained_disc_sd)
                    if rank == 0:
                        print("[INFO] Discriminator inherited from --pretrained_model (adversarial training continues)", flush=True)
                except Exception as e:
                    if rank == 0:
                        print(f"[WARNING] Could not inherit discriminator (architecture mismatch?); initializing from scratch: {e}", flush=True)
            optimizer_d = optim.AdamW(disc.parameters(), lr=args.disc_lr,
                                      betas=(0.0, 0.9), weight_decay=0.0, fused=False)
            scaler_d = GradScaler('cuda')
            if rank == 0:
                nd = sum(p.numel() for p in disc.parameters()) / 1e6
                print(f"[INFO] GAN enabled: discriminator {nd:.1f}M | disc_lr={args.disc_lr:.2e} | "
                      f"lambda_adv={args.lambda_adv} lambda_pool={args.lambda_pool} | "
                      f"warm-up {args.disc_warmup_epochs} epoch(s)", flush=True)

        start_epoch = 1
        if args.resume_from and os.path.exists(args.resume_from):
            if rank == 0:
                print(f"[INFO] Resuming from checkpoint: {args.resume_from}", flush=True)
            # Every rank loads it so weights stay identical across GPUs
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
            unwrap(model).load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Restore the discriminator (if present in the checkpoint and GAN is enabled)
            if args.gan and disc is not None and checkpoint.get('disc_state_dict'):
                disc.load_state_dict(checkpoint['disc_state_dict'])
                if checkpoint.get('disc_optimizer_state_dict'):
                    optimizer_d.load_state_dict(checkpoint['disc_optimizer_state_dict'])
                if rank == 0:
                    print("[INFO] Discriminator restored from checkpoint", flush=True)
            # Prefer the checkpoint's data_max if present (overrides the scan result)
            if 'data_max' in checkpoint and checkpoint['data_max'] is not None:
                args.data_max = float(checkpoint['data_max'])
                args.heavy_rain_threshold = 32.0 / args.data_max
            if rank == 0:
                print(f"[INFO] Resuming at epoch {start_epoch}, data_max={args.data_max:.4f}", flush=True)

        loss_history = []
        best_csi = -1.0

        for epoch in range(start_epoch, args.epochs + 1):
            if exiter.should_exit:
                if rank == 0:
                    print("[INFO] Exit signal received; saving the model...", flush=True)
                break

            epoch_start = time.time()

            sampler.set_epoch(epoch)

            # Discriminator warm-up: for the first disc_warmup_epochs, train D only
            adv_active = args.gan and (epoch > args.disc_warmup_epochs)
            avg_loss = train_one_epoch(
                model, train_loader, optimizer, scaler,
                device, args, grad_loss_fn, rank,
                disc=disc, optimizer_d=optimizer_d, scaler_d=scaler_d,
                world_size=world_size, adv_active=adv_active,
            )

            epoch_time = time.time() - epoch_start

            loss_history.append(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # -- Validation: compute CSI on the held-out set every val_interval epochs --
            run_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)
            csi_dict, mean_csi = {20.0: 0.0, 30.0: 0.0, 40.0: 0.0}, -1.0
            if run_val:
                if rank == 0:
                    csi_dict, mean_csi = evaluate_csi(
                        unwrap(model), val_dataset, device, args, args.val_max_samples
                    )
                # Broadcast CSI to all ranks so the scheduler/LR stay consistent
                csi_tensor = torch.tensor(
                    [mean_csi, csi_dict[20.0], csi_dict[30.0], csi_dict[40.0]],
                    device=device, dtype=torch.float32,
                )
                if world_size > 1:
                    dist.broadcast(csi_tensor, src=0)
                mean_csi = csi_tensor[0].item()
                csi_dict = {20.0: csi_tensor[1].item(),
                            30.0: csi_tensor[2].item(),
                            40.0: csi_tensor[3].item()}

            if rank == 0:
                val_str = ""
                if run_val:
                    val_str = (f" | CSI(mean/20/30/40): {mean_csi:.4f}/"
                               f"{csi_dict[20.0]:.4f}/{csi_dict[30.0]:.4f}/{csi_dict[40.0]:.4f}")
                print(f"[Epoch {epoch:3d}/{args.epochs}] Loss: {avg_loss:.6f} | "
                      f"LR: {current_lr:.2e} | elapsed: {epoch_time/3600:.2f}h "
                      f"({epoch_time:.1f}s){val_str}", flush=True)

                append_csv_log(
                    args.log_file, epoch, avg_loss, current_lr, epoch_time,
                    mean_csi if run_val else float('nan'),
                    csi_dict[20.0], csi_dict[30.0], csi_dict[40.0], rank
                )

            # Step the scheduler on CSI only during validation epochs (mode='max')
            if run_val and math.isfinite(mean_csi) and mean_csi >= 0:
                scheduler.step(mean_csi)

            new_lr = optimizer.param_groups[0]['lr']
            if rank == 0 and new_lr < current_lr:
                print(f"  [LR] reduced: {current_lr:.2e} -> {new_lr:.2e}", flush=True)

            if rank == 0:
                raw_model = unwrap(model)

                def _save(path):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'disc_state_dict': disc.state_dict() if disc is not None else None,
                        'disc_optimizer_state_dict': optimizer_d.state_dict() if optimizer_d is not None else None,
                        'loss': avg_loss,
                        'val_csi': mean_csi,
                        'data_max': args.data_max,
                        'args': args,
                    }, path)

                if epoch % 5 == 0 or epoch == args.epochs:
                    checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.ckpt')
                    _save(checkpoint_path)
                    print(f"[INFO] Checkpoint saved: {checkpoint_path}", flush=True)

                # Save the best model by validation CSI
                if run_val and mean_csi > best_csi:
                    best_csi = mean_csi
                    _save(os.path.join(args.save_dir, 'best_model.ckpt'))
                    print(f"[INFO] New best model (CSI={best_csi:.4f})", flush=True)

                latest_path = os.path.join(args.save_dir, 'latest_model.ckpt')
                torch.save(raw_model.state_dict(), latest_path)

            if epoch % args.cleanup_interval == 0:
                mem_monitor.cleanup()

        if rank == 0:
            print("="*60, flush=True)
            print("[INFO] Training complete.", flush=True)
            if loss_history:
                print(f"[INFO] Final loss: {loss_history[-1]:.6f}", flush=True)
                print(f"[INFO] Best validation CSI: {best_csi:.4f}", flush=True)
            print(f"[INFO] Log saved to: {args.log_file}", flush=True)
            print("="*60, flush=True)
                
    except Exception as e:
        if rank == 0:
            print(f"\n[ERROR] Training failed: {e}", flush=True)
            traceback.print_exc()
        
        try:
            if rank == 0 and 'model' in locals():
                emergency_path = os.path.join(args.save_dir, f'emergency_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ckpt')
                raw_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch if 'epoch' in locals() else 0,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
                }, emergency_path)
                print(f"[INFO] Emergency checkpoint saved: {emergency_path}", flush=True)
        except:
            pass
        
        cleanup_distributed()
        sys.exit(1)
    
    cleanup_distributed()

if __name__ == '__main__':
    main()