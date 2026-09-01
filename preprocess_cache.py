#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One-off preprocessing: convert all NetCDF files into .npy caches that are already
resized, converted to dBZ and y-flipped.

During training, RadarTrainDataset(cache_dir=...) memory-maps these .npy files directly,
skipping the slowest bottleneck (opening NetCDF + running cv2.resize on 50 frames each),
which can cut an epoch from hours down to minutes.

The preprocessing matches train_loader._load_frames' non-cached path exactly:
  take the first variable -> /10 (0.1 dBZ -> dBZ) -> data[:, ::-1, :] flip y -> resize to (W,H).
Missing values / NaN are preserved as-is (the mask is computed later in __getitem__).

Usage:
    python preprocess_cache.py \
        --data_path /path/to/radar_data/region_A/ \
        --cache_dir /path/to/radar_data/cache_512/ \
        --img_height 512 --img_width 512 --workers 8

Note: caches are stored as float16, roughly (T x 512 x 512 x 2) bytes per file. Check disk
space first (~90 GB for 728 files at ~240 frames each). Use --dtype float32 or fewer files
if space is tight.
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
        data = data[:, ::-1, :]           # flip y axis (matches the loader)

        if data.shape[1] != img_h or data.shape[2] != img_w:
            data = np.stack([
                cv2.resize(f, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                for f in data
            ])

        data = np.ascontiguousarray(data.astype(dtype))
        tmp = out_path + '.tmp.npy'
        np.save(tmp, data)
        os.replace(tmp, out_path)         # atomic replace, avoids half-written files
        return out_path, f'ok {data.shape}'
    except Exception as e:
        return out_path, f'ERROR: {e}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', required=True, help='directory containing .nc files (recursive)')
    p.add_argument('--cache_dir', required=True, help='output directory for .npy caches')
    p.add_argument('--img_height', type=int, default=512)
    p.add_argument('--img_width', type=int, default=512)
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--overwrite', action='store_true', help='regenerate even if the cache already exists')
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    nc_files = _collect_nc_files(args.data_path)
    if not nc_files:
        print(f'[ERROR] No NetCDF files found: {args.data_path}', flush=True)
        return
    print(f'[INFO] {len(nc_files)} NetCDF files -> {args.cache_dir} '
          f'(dtype={args.dtype}, {args.img_height}x{args.img_width})', flush=True)

    dtype = np.float16 if args.dtype == 'float16' else np.float32
    done = 0
    errors = 0

    from functools import partial
    task = partial(convert_one, cache_dir=args.cache_dir,
                   img_h=args.img_height, img_w=args.img_width,
                   dtype=dtype, overwrite=args.overwrite)

    # Multiprocessing: each process opens NetCDF independently, sidestepping the
    # thread-safety problems of HDF5/netCDF4
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (out, status) in enumerate(ex.map(task, nc_files), 1):
            if status.startswith('ERROR'):
                errors += 1
                print(f'  [{i}/{len(nc_files)}] {os.path.basename(out)} {status}', flush=True)
            done += 1
            if i % 50 == 0 or i == len(nc_files):
                print(f'  [{i}/{len(nc_files)}] processed ({errors} errors)', flush=True)

    print(f'[DONE] {done} files done, {errors} errors. Cache dir: {args.cache_dir}', flush=True)


if __name__ == '__main__':
    main()
