#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一次性预处理：把所有 NC 文件转成已 resize/转 dBZ/翻转 y 轴的 .npy 缓存。

训练时 RadarTrainDataset(cache_dir=...) 会直接 mmap 读取这些 .npy，
跳过「开 NC 文件 + 对 50 帧逐帧 cv2.resize」这个最慢的瓶颈，
每 epoch 可从数小时压到分钟级。

预处理逻辑与 train_loader._load_frames 的非缓存路径完全一致：
  取首个变量 → /10（0.1dBZ→dBZ）→ data[:, ::-1, :] 翻转 y → resize 到 (W,H)。
缺测/NaN 原样保留（mask 在 __getitem__ 里再算）。

用法：
    python preprocess_cache.py \
        --data_path /workspace/tmp/tanch/data/radar/training_data_Pek/ \
        --cache_dir /workspace/tmp/tanch/data/radar/cache_512/ \
        --img_height 512 --img_width 512 --workers 8

注意：缓存按 float16 存，约 (T×512×512×2) 字节/文件。先确认磁盘空间足够
（728 个文件、每文件 ~240 帧 时约 90GB）。空间紧张可加 --dtype float32 或减少文件。
"""
import os
import sys
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nowcasting.data_provider.train_loader import _collect_nc_files, nc_to_cache_name

import cv2


def convert_one(nc_path, cache_dir, img_h, img_w, dtype, overwrite):
    out_path = os.path.join(cache_dir, nc_to_cache_name(nc_path))
    if os.path.exists(out_path) and not overwrite:
        return out_path, 'skip(exists)'
    try:
        import xarray as xr
        with xr.open_dataset(nc_path, decode_times=False) as ds:
            var_names = list(ds.data_vars)
            if not var_names:
                return out_path, 'skip(no var)'
            raw = ds[var_names[0]].values
        if raw.ndim == 2:
            raw = raw[np.newaxis, ...]
        if raw.ndim < 2:
            return out_path, 'skip(bad ndim)'

        data = raw.astype(np.float32)
        data = data / 10.0                # 0.1 dBZ → dBZ
        data = data[:, ::-1, :]           # 翻转 y 轴（与 loader 一致）

        if data.shape[1] != img_h or data.shape[2] != img_w:
            data = np.stack([
                cv2.resize(f, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                for f in data
            ])

        data = np.ascontiguousarray(data.astype(dtype))
        tmp = out_path + '.tmp.npy'
        np.save(tmp, data)
        os.replace(tmp, out_path)         # 原子替换，避免半截文件
        return out_path, f'ok {data.shape}'
    except Exception as e:
        return out_path, f'ERROR: {e}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', required=True, help='含 .nc 的目录（支持多级）')
    p.add_argument('--cache_dir', required=True, help='输出 .npy 缓存目录')
    p.add_argument('--img_height', type=int, default=512)
    p.add_argument('--img_width', type=int, default=512)
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--overwrite', action='store_true', help='已存在也重新生成')
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    nc_files = _collect_nc_files(args.data_path)
    if not nc_files:
        print(f'[ERROR] 未找到 NC 文件: {args.data_path}', flush=True)
        return
    print(f'[INFO] 共 {len(nc_files)} 个 NC 文件 → {args.cache_dir} '
          f'(dtype={args.dtype}, {args.img_height}x{args.img_width})', flush=True)

    dtype = np.float16 if args.dtype == 'float16' else np.float32
    done = 0
    errors = 0

    from functools import partial
    task = partial(convert_one, cache_dir=args.cache_dir,
                   img_h=args.img_height, img_w=args.img_width,
                   dtype=dtype, overwrite=args.overwrite)

    # 多进程：每个进程独立打开 netCDF，绕开 HDF5/netCDF4 非线程安全问题
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (out, status) in enumerate(ex.map(task, nc_files), 1):
            if status.startswith('ERROR'):
                errors += 1
                print(f'  [{i}/{len(nc_files)}] {os.path.basename(out)} {status}', flush=True)
            done += 1
            if i % 50 == 0 or i == len(nc_files):
                print(f'  [{i}/{len(nc_files)}] 已处理（错误 {errors}）', flush=True)

    print(f'[DONE] 完成 {done} 个，错误 {errors} 个。缓存目录: {args.cache_dir}', flush=True)


if __name__ == '__main__':
    main()
