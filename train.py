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

# 华北+华南混合训练：逗号分隔多个目录（不写 --train_data_path 时默认用这个）
TRAIN_DATA_PATH = '/path/to/radar_data/region_A,/path/to/radar_data/region_B'
CACHE_DIR = '/path/to/radar_data/cache_512/'
SAVE_DIR = './checkpoints'
LOG_FILE = './checkpoints/training_log.csv'

class ParallelBatchLoader:
    """
    線程版並行數據加載器，完全繞過 /dev/shm 限制。
    多進程 DataLoader worker 需要通過共享內存傳輸 tensor（Docker 默認 64MB 不夠）。
    改用線程後，所有線程共享主進程內存，無 shm 開銷。
    xarray 讀取 NC 文件時釋放 GIL，多線程可真正並行 IO。
    """
    def __init__(self, dataset, sampler, batch_size, num_threads=8, prefetch=2):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.prefetch = prefetch
        # 仅当使用 .npy 缓存（mmap，线程安全）时才并行读取 batch 内样本；
        # 回退到原始 netCDF 路径时必须顺序读（HDF5/netCDF4 非线程安全，并发会 SIGSEGV）。
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
            # 缓存模式：多线程并行读（mmap 读 + numpy 拷贝会释放 GIL），喂满 GPU
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
        print(f"\n[INFO] 收到信号 {signum}，正在保存模型并退出...", flush=True)
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
    DGMR / NowcastNet 风格的分级前景加权（基于真实 dBZ 阈值）。
    雷达场 99%+ 像素是 0/弱回波，必须给强回波极高权重，
    否则模型会塌缩到「输出空白」这个平凡解（loss 最小但毫无预报价值）。
    """
    dbz = target_norm * data_max
    w = torch.ones_like(dbz)
    w = torch.where(dbz >= 10.0, torch.full_like(w, 2.0), w)
    w = torch.where(dbz >= 20.0, torch.full_like(w, 3.0), w)
    w = torch.where(dbz >= 30.0, torch.full_like(w, 5.0), w)
    w = torch.where(dbz >= 40.0, torch.full_like(w, 10.0), w)  # 上限 30→10，缓解小batch梯度尖峰
    return w

def balanced_l1(pred, target, mask, data_max, fg_thresh_dbz=15.0, bg_weight=0.2):
    """频率平衡 L1：前景 / 背景分别求均值再相加。

    关键点：绝不对「全体像素」求加权平均——雷达场 99.99% 是零背景，
    任何 sum/总数 形式的归一化都会把前景误差稀释到 ~1e-5，导致模型塌缩到
    「输出空白」。这里前景单独按自身像素数求均值，每个强回波像素都拿到
    强梯度；背景以小权重(bg_weight)参与，避免满屏噪声。前景内部再按
    dBZ 分级加权(radar_weight)，强回波惩罚更重。"""
    diff = (pred - target).abs() * mask
    w = radar_weight(target, data_max)
    dbz = target * data_max
    fg = (dbz >= fg_thresh_dbz).float() * mask
    bg = (dbz < fg_thresh_dbz).float() * mask

    fg_loss = (w * diff * fg).sum() / (fg.sum() + 1e-6)
    bg_loss = (diff * bg).sum() / (bg.sum() + 1e-6)
    return fg_loss + bg_weight * bg_loss

def motion_smoothness(motion):
    """光流场空间平滑正则：motion (B, T, 2, H, W)。
    约束 evolution 网络学出连续、物理合理的平流场，而非杂乱位移。"""
    b, t, c, h, w = motion.shape
    m = motion.reshape(b * t, c, h, w)
    dx = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
    dy = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
    return dx + dy


@torch.no_grad()
def evaluate_csi(raw_model, val_dataset, device, args, max_samples=200):
    """在留出验证集上计算 CSI（临界成功指数），这是临近预报的标准评分。
    空白预报的 CSI=0，所以它能直接戳穿「loss 在降但模型没用」的假象。
    返回 (各阈值CSI字典, 平均CSI)。仅在 rank 0 调用。"""
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
    """训练一个 epoch。
    - disc=None：纯 L1 训练（原行为，完全不变）。
    - disc!=None：交替对抗训练（GAN）。判别器 disc 为「原始模块」(不套 DDP)，
      多卡时手动 all-reduce 其梯度保持同步；生成器 model 仍是 DDP。
    - adv_active=False：判别器热身阶段，D 照常训练，但暂不把对抗损失加到 G。
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
                print("[INFO] 收到退出信号，停止训练...", flush=True)
            break

        try:
            frames = frames.to(device, non_blocking=True)
            target = frames[:, args.input_length:, :, :, 0]
            mask = frames[:, args.input_length:, :, :, 1]

            # ── 生成器前向 + 基础 L1/结构/演变损失 ──────────────────────────
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

            # ── GAN：判别器一步 + 生成器对抗项 ─────────────────────────────
            # 判别器全程跑 float32、不用 autocast/scaler_d，彻底避开 AMP+GAN 的
            # "unscale FP16 gradients" 冲突。判别器小(1.2M)，float32 开销可忽略。
            if disc is not None:
                input_seq = frames[:, :args.input_length, :, :, 0].float()   # (B, IL, H, W)
                real_seq = frames[:, :, :, :, 0].float()                     # (B, total, H, W)
                fake_seq = torch.cat([input_seq, pred.float()], dim=1)       # (B, total, H, W)

                # (1) 训练判别器（float32，普通 backward，无 scaler）
                optimizer_d.zero_grad(set_to_none=True)
                d_real = disc(real_seq)
                d_fake = disc(fake_seq.detach())
                d_loss = hinge_loss_d(d_real, d_fake)
                d_loss.backward()
                if dist.is_initialized():                                    # 手动同步 D 梯度
                    for p in disc.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(world_size)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                optimizer_d.step()
                dv = d_loss.item()
                running_d += dv if math.isfinite(dv) else 0.0

                # (2) 生成器对抗项：冻结 D 参数（梯度仍能流回 G），加到 g_loss。
                #     D 保持冻结到 G 的 backward 做完（在下面 G 步之后再解冻）。
                if adv_active:
                    for p in disc.parameters():
                        p.requires_grad_(False)
                    g_adv = hinge_loss_g(disc(fake_seq))                      # float32
                    pool = pool_regularization(pred.float(), target.float())
                    g_loss = g_loss + args.lambda_adv * g_adv + args.lambda_pool * pool

            # ── 生成器一步（所有 rank 都 backward，避免 DDP 死锁）──────────
            scaler.scale(g_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            # G 步做完后再解冻 D，供下一个 batch 训练判别器
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
                          f"(loss={loss_val})，GradScaler 已自动跳过该步更新", flush=True)

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
        print(f"[WARNING] 本epoch共跳过 {nan_batches} 个NaN/Inf batch", flush=True)
    if local_rank == 0 and disc is not None and valid_batches > 0:
        print(f"[GAN] 本epoch 判别器平均 loss: {running_d / max(valid_batches,1):.4f} "
              f"| 对抗项启用: {adv_active}", flush=True)

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
    # 在 NCCL 初始化前设置心跳超时，覆盖默认 480 秒
    # 避免 rank 0 扫描数据集时 rank 1/2/3 在 barrier 处触发 watchdog
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
            timeout=timedelta(hours=2),  # 数据扫描可能超过10分钟默认值
        )
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def unwrap(model):
    """取出未被 DDP 包裹的原始模型（单卡时 model 本身就是原始模型）。"""
    return model.module if hasattr(model, 'module') else model

def main():
    rank, world_size, local_rank = setup_distributed()
    
    parser = argparse.ArgumentParser(description='NowcastNet Multi-GPU Training')
    parser.add_argument('--train_data_path', type=str, default=TRAIN_DATA_PATH)
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR,
                        help='预处理 .npy 缓存目录（preprocess_cache.py 生成）。'
                             '提供后直接 mmap 读取，大幅加速数据加载。设为 none 可禁用')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--log_file', type=str, default=LOG_FILE)
    
    parser.add_argument('--batch_size', type=int, default=16, help='每张卡的batch size')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=12, help='每卡数据加载线程数')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    
    parser.add_argument('--cleanup_interval', type=int, default=20)
    parser.add_argument('--monitor_interval', type=int, default=100)
    
    parser.add_argument('--input_length', type=int, default=20)
    parser.add_argument('--total_length', type=int, default=50)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--warp_mode', type=str, default='bilinear', choices=['bilinear', 'nearest'],
                        help='evolution 迭代 warp 的插值模式。bilinear 可减轻 30 步迭代的'
                             '累积平滑(台风等精细结构保持更久)；nearest 为原版行为')
    parser.add_argument('--lambda_grad', type=float, default=0.1)
    parser.add_argument('--lambda_evo', type=float, default=1.0,
                        help='evolution 网络自监督损失权重')
    parser.add_argument('--lambda_motion', type=float, default=0.01,
                        help='光流场平滑正则权重')
    parser.add_argument('--fg_thresh_dbz', type=float, default=15.0,
                        help='前景/背景分界阈值(dBZ)，>=此值算前景单独求均值')
    parser.add_argument('--bg_weight', type=float, default=0.2,
                        help='背景损失权重，避免满屏噪声但不让背景主导')
    # ── GAN / 判别器（对抗训练，修复"模糊/强度塌缩"）──
    parser.add_argument('--gan', action='store_true',
                        help='开启对抗训练(判别器)。建议从 best_model warm start(--pretrained_model)')
    parser.add_argument('--lambda_adv', type=float, default=0.01,
                        help='对抗损失权重(生成器)。太大易崩，太小无效果，先用 0.01')
    parser.add_argument('--lambda_pool', type=float, default=1.0,
                        help='池化正则权重，约束粗尺度降水量、防止判别器逼模型乱造强回波')
    parser.add_argument('--disc_lr', type=float, default=2e-4,
                        help='判别器学习率')
    parser.add_argument('--disc_base_c', type=int, default=32,
                        help='判别器基础通道数')
    parser.add_argument('--disc_warmup_epochs', type=int, default=1,
                        help='判别器热身轮数：这几轮只训 D、不给 G 加对抗项，稳定后再开')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='按文件留出的验证集比例（时间上独立）')
    parser.add_argument('--val_interval', type=int, default=2,
                        help='每多少个 epoch 在验证集上算一次 CSI')
    parser.add_argument('--val_max_samples', type=int, default=200,
                        help='每次验证最多评估多少个样本（控制耗时）')
    parser.add_argument('--use_compile', action='store_true', default=False)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='仅加载模型权重作为起点（fine-tune），不恢复optimizer/epoch')
    parser.add_argument('--data_max', dest='data_max_fixed', type=float, default=0.0,
                        help='手动锁定归一化基准 data_max(>0生效)。换数据集做 warm start 时'
                             '务必与预训练一致(本项目为 55.5)；留空则从 pretrained_model 继承或自动统计')
    
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    
    args = parser.parse_args()

    # 关键修复：如果 num_workers=0，确保 prefetch_factor=None
    if args.num_workers == 0:
        args.prefetch_factor = None

    # 允许用 --cache_dir none 显式禁用缓存（回退到直接读 NC）
    if args.cache_dir is not None and args.cache_dir.strip().lower() in ('none', ''):
        args.cache_dir = None

    global_batch_size = args.batch_size * world_size
    if rank == 0:
        print(f"[INFO] 分布式训练: {world_size} 张GPU", flush=True)
        print(f"[INFO] 每卡Batch Size: {args.batch_size}", flush=True)
        print(f"[INFO] 全局Batch Size: {global_batch_size}", flush=True)
        print(f"[INFO] 学习率: {args.lr}", flush=True)

    args.evo_ic = args.total_length - args.input_length
    args.gen_oc = args.total_length - args.input_length
    args.ic_feature = args.ngf * 10

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        # 所有进程都需要设置，不只是rank 0
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        if rank == 0:
            print(f"[INFO] GPU: {torch.cuda.get_device_name(local_rank)}", flush=True)
            print(f"[INFO] 单卡显存: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f}GB", flush=True)
            print(f"[INFO] 总显存: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3 * world_size:.1f}GB", flush=True)

    mem_monitor = MemoryMonitor(device, local_rank)
    exiter = GracefulExiter()
    train_one_epoch.exiter = exiter

    if rank == 0:
        print("="*60, flush=True)
        print(f"[INFO] 开始分布式训练任务", flush=True)
        print(f"[INFO] 每卡Batch Size: {args.batch_size}", flush=True)
        print(f"[INFO] 梯度累积步数: {args.grad_accum_steps}", flush=True)
        print(f"[INFO] 初始学习率: {args.lr}", flush=True)
        print(f"[INFO] LR 衰减因子: {args.lr_factor}", flush=True)
        print(f"[INFO] LR 耐心值: {args.lr_patience}", flush=True)
        print(f"[INFO] 最小学习率: {args.lr_min}", flush=True)
        print(f"[INFO] Workers每卡: {args.num_workers}", flush=True)
        print(f"[INFO] 日志文件: {args.log_file}", flush=True)
        print("="*60, flush=True)

    init_csv_log(args.log_file, rank)

    try:
        # ── Step 1: 确定归一化基准 data_max ────────────────────────────────
        # 优先级：命令行 --data_max > --pretrained_model 内含的 > 自动统计。
        # warm start（尤其换数据集时）必须沿用预训练的 data_max，否则归一化尺度
        # 一变，已训好的权重全部对不上，等于白训。
        cache_file = os.path.join(args.save_dir, 'data_stats.json')

        pinned, src = None, ''
        if getattr(args, 'data_max_fixed', 0) and args.data_max_fixed > 0:
            pinned, src = float(args.data_max_fixed), '命令行 --data_max'
        elif args.pretrained_model and os.path.exists(args.pretrained_model):
            try:
                _c = torch.load(args.pretrained_model, map_location='cpu', weights_only=False)
                if isinstance(_c, dict) and _c.get('data_max'):
                    pinned, src = float(_c['data_max']), 'pretrained_model 继承'
                del _c
            except Exception:
                pinned = None

        if pinned is not None:
            data_max = pinned
            if rank == 0:
                print(f"[INFO] 使用固定 data_max={data_max:.4f}（{src}，跳过统计扫描）", flush=True)
        else:
            if rank == 0:
                data_max = compute_data_stats(args.train_data_path, percentile=99.9, cache_file=cache_file)
            # barrier 等 rank 0 写完缓存文件，其余 rank 再读
            if world_size > 1:
                dist.barrier()
            if rank != 0:
                with open(cache_file, 'r') as f:
                    data_max = float(json.load(f)['data_max'])

        args.data_max = data_max
        # 强回波权重阈值：对应原始空间约32单位（与data_max成比例）
        args.heavy_rain_threshold = 32.0 / args.data_max

        if rank == 0:
            print(f"[INFO] data_max={args.data_max:.4f}, 强回波阈值(归一化)={args.heavy_rain_threshold:.4f}", flush=True)

        # ── Step 2: 加载数据集（仅 rank 0 扫描磁盘，再广播给其他 rank）────────
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
            print("[INFO] 正在加载数据集（rank 0 扫描）...", flush=True)
            if os.path.exists(samples_cache_file):
                with open(samples_cache_file, 'rb') as f:
                    all_samples = pickle.load(f)
                print(f"[INFO] 从缓存加载样本索引，共 {len(all_samples)} 个样本", flush=True)
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
                print(f"[INFO] 样本索引已缓存至: {samples_cache_file}", flush=True)

            # ── 按文件留出验证集（时间上独立，避免训练/验证泄漏）──────────────
            files = sorted(set(s[0] for s in all_samples))
            n_val_files = max(1, int(round(len(files) * args.val_fraction)))
            val_files = set(files[-n_val_files:])
            train_samples = [s for s in all_samples if s[0] not in val_files]
            val_samples = [s for s in all_samples if s[0] in val_files]
            load_time = time.time() - start_time
            print(f"[INFO] 数据集初始化完成，耗时 {load_time:.1f}s", flush=True)
            print(f"[INFO] 训练样本 {len(train_samples)} 个（{len(files)-n_val_files} 文件），"
                  f"验证样本 {len(val_samples)} 个（{n_val_files} 文件）", flush=True)
            samples_to_broadcast = [train_samples, val_samples]
        else:
            samples_to_broadcast = [None, None]

        # 将训练样本索引广播给其他 rank（验证只在 rank 0 进行）
        if world_size > 1:
            dist.broadcast_object_list(samples_to_broadcast, src=0)

        dataset = _build_dataset(samples_to_broadcast[0])

        # 验证集仅 rank 0 构建
        val_dataset = _build_dataset(samples_to_broadcast[1]) if rank == 0 else None

        # 所有进程同步
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
            # 線程版加載器：繞過 Docker /dev/shm 限制，IO 並行不需要共享內存
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
            print(f"[INFO] DataLoader 创建完成，每卡约 {len(train_loader)} 个batch", flush=True)

        if rank == 0:
            print("[INFO] 正在初始化模型...", flush=True)
        model = Net(args).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if rank == 0:
            print(f"[INFO] 模型参数: 总计 {total_params/1e6:.1f}M, 可训练 {trainable_params/1e6:.1f}M", flush=True)
        
        # 加载预训练权重（fine-tune 起点，不恢复 optimizer/epoch）
        pretrained_disc_sd = None   # 若 ckpt 里带判别器，稍后创建 disc 时一并继承
        if args.pretrained_model and os.path.exists(args.pretrained_model):
            if rank == 0:
                print(f"[INFO] 加载预训练权重: {args.pretrained_model}", flush=True)
            ckpt = torch.load(args.pretrained_model, map_location=device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if isinstance(ckpt, dict):
                pretrained_disc_sd = ckpt.get('disc_state_dict', None)
            if rank == 0:
                print(f"[INFO] 预训练权重加载完成 (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

        if args.use_compile and hasattr(torch, 'compile'):
            if rank == 0:
                print("[INFO] 正在应用 torch.compile...", flush=True)
            model = torch.compile(model)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        elif rank == 0:
            print("[INFO] 单卡训练，跳过 DDP 包裹", flush=True)

        scaled_lr = args.lr * math.sqrt(world_size)
        if rank == 0:
            print(f"[INFO] 缩放后学习率: {scaled_lr:.2e} (原 {args.lr:.2e} * sqrt({world_size}))", flush=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=1e-5, fused=False)
        
        # 以验证集 CSI 为调度依据（越大越好）。CSI=0 时空白预报无法蒙混过关，
        # 比监控 train_loss 可靠得多（旧方案在 loss≈1e-5 时阈值失效，LR 永不下降）。
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )
        
        scaler = GradScaler('cuda')
        grad_loss_fn = SobelGradientLoss().to(device)

        # ── GAN：判别器（原始模块，不套 DDP；多卡时在 train_one_epoch 里手动同步梯度）──
        disc = optimizer_d = scaler_d = None
        if args.gan:
            disc = Temporal_Discriminator(args.total_length, base_c=args.disc_base_c).to(device)
            # 若预训练 ckpt 里带判别器（上一轮 GAN 训练的产物），一并继承，避免 D 从零重来
            if pretrained_disc_sd is not None:
                try:
                    disc.load_state_dict(pretrained_disc_sd)
                    if rank == 0:
                        print("[INFO] 判别器已从 --pretrained_model 继承（继续对抗，不从零开始）", flush=True)
                except Exception as e:
                    if rank == 0:
                        print(f"[WARNING] 判别器继承失败（结构不匹配？），从零初始化: {e}", flush=True)
            optimizer_d = optim.AdamW(disc.parameters(), lr=args.disc_lr,
                                      betas=(0.0, 0.9), weight_decay=0.0, fused=False)
            scaler_d = GradScaler('cuda')
            if rank == 0:
                nd = sum(p.numel() for p in disc.parameters()) / 1e6
                print(f"[INFO] GAN 开启：判别器 {nd:.1f}M | disc_lr={args.disc_lr:.2e} | "
                      f"lambda_adv={args.lambda_adv} lambda_pool={args.lambda_pool} | "
                      f"热身 {args.disc_warmup_epochs} 轮", flush=True)

        start_epoch = 1
        if args.resume_from and os.path.exists(args.resume_from):
            if rank == 0:
                print(f"[INFO] 从检查点恢复: {args.resume_from}", flush=True)
            # 所有rank都加载，保证多卡权重一致
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
            unwrap(model).load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # 恢复判别器（若 checkpoint 里有且本次开启了 GAN）
            if args.gan and disc is not None and checkpoint.get('disc_state_dict'):
                disc.load_state_dict(checkpoint['disc_state_dict'])
                if checkpoint.get('disc_optimizer_state_dict'):
                    optimizer_d.load_state_dict(checkpoint['disc_optimizer_state_dict'])
                if rank == 0:
                    print("[INFO] 判别器已从检查点恢复", flush=True)
            # 如果checkpoint保存了data_max，优先使用（覆盖扫描结果）
            if 'data_max' in checkpoint and checkpoint['data_max'] is not None:
                args.data_max = float(checkpoint['data_max'])
                args.heavy_rain_threshold = 32.0 / args.data_max
            if rank == 0:
                print(f"[INFO] 从 epoch {start_epoch} 继续训练，data_max={args.data_max:.4f}", flush=True)

        loss_history = []
        best_csi = -1.0

        for epoch in range(start_epoch, args.epochs + 1):
            if exiter.should_exit:
                if rank == 0:
                    print("[INFO] 收到退出信号，保存模型...", flush=True)
                break

            epoch_start = time.time()

            sampler.set_epoch(epoch)

            # 判别器热身：前 disc_warmup_epochs 轮只训 D，不给 G 加对抗项
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

            # ── 验证：每 val_interval 个 epoch 在留出集上算 CSI ──────────────
            run_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)
            csi_dict, mean_csi = {20.0: 0.0, 30.0: 0.0, 40.0: 0.0}, -1.0
            if run_val:
                if rank == 0:
                    csi_dict, mean_csi = evaluate_csi(
                        unwrap(model), val_dataset, device, args, args.val_max_samples
                    )
                # 把 CSI 广播给所有 rank，保证调度器/LR 在多卡间一致
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
                      f"LR: {current_lr:.2e} | 耗时: {epoch_time/3600:.2f}h "
                      f"({epoch_time:.1f}s){val_str}", flush=True)

                append_csv_log(
                    args.log_file, epoch, avg_loss, current_lr, epoch_time,
                    mean_csi if run_val else float('nan'),
                    csi_dict[20.0], csi_dict[30.0], csi_dict[40.0], rank
                )

            # 仅在验证轮用 CSI 推进调度器（mode='max'）
            if run_val and math.isfinite(mean_csi) and mean_csi >= 0:
                scheduler.step(mean_csi)

            new_lr = optimizer.param_groups[0]['lr']
            if rank == 0 and new_lr < current_lr:
                print(f"  📉 学习率已降低: {current_lr:.2e} → {new_lr:.2e}", flush=True)

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
                    print(f"[INFO] 检查点已保存: {checkpoint_path}", flush=True)

                # 按验证 CSI 保存最优模型
                if run_val and mean_csi > best_csi:
                    best_csi = mean_csi
                    _save(os.path.join(args.save_dir, 'best_model.ckpt'))
                    print(f"[INFO] 🏆 新的最优模型 (CSI={best_csi:.4f})", flush=True)

                latest_path = os.path.join(args.save_dir, 'latest_model.ckpt')
                torch.save(raw_model.state_dict(), latest_path)

            if epoch % args.cleanup_interval == 0:
                mem_monitor.cleanup()

        if rank == 0:
            print("="*60, flush=True)
            print("[INFO] 训练完成！", flush=True)
            if loss_history:
                print(f"[INFO] 最终 Loss: {loss_history[-1]:.6f}", flush=True)
                print(f"[INFO] 最优验证 CSI: {best_csi:.4f}", flush=True)
            print(f"[INFO] 日志已保存到: {args.log_file}", flush=True)
            print("="*60, flush=True)
                
    except Exception as e:
        if rank == 0:
            print(f"\n[ERROR] 训练发生错误: {e}", flush=True)
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
                print(f"[INFO] 紧急检查点已保存: {emergency_path}", flush=True)
        except:
            pass
        
        cleanup_distributed()
        sys.exit(1)
    
    cleanup_distributed()

if __name__ == '__main__':
    main()