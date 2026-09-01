import os
import json
import threading
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


def compute_data_stats(data_path, percentile=99.9, cache_file=None):
    """
    Scan all training NetCDF files and use the given percentile as the normalization
    reference data_max. The percentile is computed per file, and the maximum across all
    files becomes the global data_max.

    Args:
        data_path   : directory of NetCDF files, or a single NetCDF path
        percentile  : percentile to use (default 99.9, more robust than the global max)
        cache_file  : JSON cache path; if it exists it is read directly, skipping the scan

    Returns:
        data_max (float)
    """
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            stats = json.load(f)
        print(f"[Stats] data_max={stats['data_max']:.4f} read from cache (p{stats['percentile']}, {stats['n_files']} files)", flush=True)
        return float(stats['data_max'])

    nc_files = _collect_nc_files(data_path)
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found: {data_path}")

    print(f"[Stats] Scanning {len(nc_files)} files, computing p{percentile}...", flush=True)
    per_file_pct = []

    for i, nc_path in enumerate(nc_files):
        try:
            import xarray as xr
            with xr.open_dataset(nc_path, decode_times=False) as ds:
                var_names = list(ds.data_vars)
                if not var_names:
                    continue
                data = ds[var_names[0]].values.astype(np.float32)
            # Unit is 0.1 dBZ; convert to true dBZ first
            data = data / 10.0
            valid = data[np.isfinite(data) & (data >= 0)]
            if valid.size > 0:
                per_file_pct.append(float(np.percentile(valid, percentile)))
        except Exception as e:
            print(f"  Warning: skipping {os.path.basename(nc_path)}: {e}", flush=True)

        if (i + 1) % 100 == 0 or (i + 1) == len(nc_files):
            cur_max = max(per_file_pct) if per_file_pct else 0.0
            print(f"  [{i+1}/{len(nc_files)}] current max p{percentile}: {cur_max:.4f}", flush=True)

    if not per_file_pct:
        raise ValueError("Could not compute statistics from the training data; check the NetCDF format")

    data_max = float(np.max(per_file_pct))
    print(f"[Stats] Scan complete: data_max={data_max:.4f} (p{percentile}, {len(per_file_pct)} valid files)", flush=True)

    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({'data_max': data_max, 'percentile': percentile, 'n_files': len(per_file_pct)}, f, indent=2)
        print(f"[Stats] Statistics cached to: {cache_file}", flush=True)

    return data_max


def nc_to_cache_name(nc_path):
    """Map a NetCDF file's full path to a unique cache filename (so identically named files
    in different subdirectories do not collide). The conversion script and the Dataset must
    use the same mapping."""
    key = os.path.abspath(nc_path).replace(os.sep, '__').lstrip('_')
    return key + '.npy'


def _collect_nc_files(path):
    """
    Recursively collect all .nc files under path (or path itself) and return a sorted list.
    path may be:
      - a single .nc file path
      - a directory containing .nc files (nested subdirectories supported)
      - several comma-separated directories (e.g. mixed multi-region training:
        "/data/radar/region_A,/data/radar/region_B")
    """
    # Comma-separated multi-path: collect each and merge (mixed multi-region training)
    if isinstance(path, str) and ',' in path:
        merged = []
        for p in path.split(','):
            p = p.strip()
            if p:
                merged.extend(_collect_nc_files(p))
        return sorted(merged)

    if os.path.isfile(path):
        if path.endswith('.nc'):
            return [path]
        raise ValueError(f"{path} is not a .nc file")

    nc_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.nc'):
                nc_files.append(os.path.join(root, f))
    return sorted(nc_files)


class RadarTrainDataset(Dataset):
    """
    NowcastNet training dataset. Supports several data layouts:

      1. one large .nc file containing every time step, shape=(T_total, H, W)
      2. one .nc file per day (any number of frames; 240 is not required)
      3. nested directories organized by month/year (scanned recursively)
      4. any mixture of the above

    For each .nc file, sliding-window samples are generated with the given stride:
      sample = (nc_path, start_frame_index)
    Files with fewer than total_length frames are skipped automatically.

    Args:
        data_path   : a single .nc path, or a directory containing .nc files (recursive)
        input_length: number of input frames (default 20)
        total_length: input + forecast frames (default 50)
        stride      : sliding-window stride (default 1; 6 reduces sample count ~6x)
        img_height  : target image height (default 512)
        img_width   : target image width (default 512)
    """

    def __init__(self, data_path, input_length=9, total_length=29,
                 stride=1, img_height=512, img_width=512, data_max=80.0,
                 _prebuilt_samples=None, cache_dir=None):
        self.input_length = input_length
        self.total_length = total_length
        self.img_height   = img_height
        self.img_width    = img_width
        self.data_max     = data_max
        # Preprocessed cache directory (one .npy per NetCDF, already resized/dBZ/flipped).
        # When provided and present, it is memory-mapped directly, skipping the
        # open-NetCDF + cv2.resize bottleneck.
        self.cache_dir    = cache_dir
        # netCDF/HDF5 is not thread-safe: serialize NetCDF reads with a lock to avoid
        # concurrent SIGSEGV. Cached (.npy mmap) reads bypass this lock and stay parallel.
        self._nc_lock     = threading.Lock()

        # If an index was built externally (broadcast from rank 0), use it and skip the disk scan
        if _prebuilt_samples is not None:
            self.samples = _prebuilt_samples
            return

        nc_files = _collect_nc_files(data_path)
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found under {data_path}")

        # Build the sample index: [(nc_path, var_name, start_frame_idx), ...]
        self.samples  = []
        skipped_files = 0

        for nc_path in nc_files:
            var_name, n_frames = self._probe_file(nc_path)
            if var_name is None or n_frames < total_length:
                skipped_files += 1
                continue
            # Store the variable name in the index to avoid re-probing
            for start in range(0, n_frames - total_length + 1, stride):
                self.samples.append((nc_path, var_name, start))

        print(
            f"[Dataset] Scanned {len(nc_files)} files, "
            f"skipped {skipped_files} (fewer than {total_length} frames), "
            f"generated {len(self.samples)} samples (stride={stride})"
        )

    # -- Internal helpers ------------------------------------------------------

    def _probe_file(self, nc_path):
        """
        Open a NetCDF file and return (var_name, n_frames).
        Returns (None, 0) on failure or if the file has no variables.
        """
        try:
            import xarray as xr
            with xr.open_dataset(nc_path, decode_times=False) as ds:
                var_names = list(ds.data_vars)
                if not var_names:
                    return None, 0
                var = var_names[0]
                values = ds[var].values
                # Handle varying dimensionality: treat the first axis as time
                if values.ndim < 2:
                    return None, 0
                if values.ndim == 2:
                    # Single-frame file (H, W) -> treat as 1 frame
                    return var, 1
                return var, int(values.shape[0])
        except Exception as e:
            print(f"  Warning: could not read {os.path.basename(nc_path)}: {e}")
            return None, 0

    def _load_frames(self, nc_path, var_name, start):
        """
        Read frames [start, start+total_length) from nc_path.
        Returns a float32 array of shape (total_length, img_height, img_width).
        """
        # -- Fast path: memory-map the preprocessed .npy cache ---------------
        if self.cache_dir is not None:
            cpath = os.path.join(self.cache_dir, nc_to_cache_name(nc_path))
            if os.path.exists(cpath):
                arr = np.load(cpath, mmap_mode='r')  # (T_full, H, W), already dBZ/flipped/resized
                data = np.asarray(arr[start:start + self.total_length], dtype=np.float32)
                # Safety net: resize if the cache size differs from the target (normally unused)
                if data.shape[1] != self.img_height or data.shape[2] != self.img_width:
                    data = np.stack([
                        cv2.resize(f, (self.img_width, self.img_height),
                                   interpolation=cv2.INTER_LINEAR)
                        for f in data
                    ])
                return data.astype(np.float32)
            # On a cache miss, fall back to reading the original NetCDF

        import xarray as xr
        # netCDF/HDF5 is not thread-safe; serialize with a lock to avoid SIGSEGV under
        # concurrent multi-threaded reads
        with self._nc_lock:
            with xr.open_dataset(nc_path, decode_times=False) as ds:
                da = ds[var_name]
                # Read only the needed frames instead of loading the whole file
                if da.ndim >= 3:
                    time_dim = da.dims[0]
                    raw = da.isel({time_dim: slice(start, start + self.total_length)}).values
                else:
                    raw = da.values  # single-frame file

        # Normalize to (T, H, W)
        if raw.ndim == 2:
            raw = raw[np.newaxis, ...]

        data = raw[:self.total_length].copy()  # (T, H, W)

        # Stored unit is 0.1 dBZ (value = dBZ x 10); convert to true dBZ
        data = data / 10.0

        # Flip the y axis (consistent with the existing loader)
        data = data[:, ::-1, :]

        # Resize to the target resolution
        if data.shape[1] != self.img_height or data.shape[2] != self.img_width:
            resized = []
            for frame in data:
                r = cv2.resize(
                    frame.astype(np.float32),
                    (self.img_width, self.img_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized.append(r)
            data = np.stack(resized)

        return data.astype(np.float32)

    # -- Dataset interface -----------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        nc_path, var_name, start = self.samples[index]
        data = self._load_frames(nc_path, var_name, start)  # (T, H, W)

        # Validity mask (missing/negative values count as invalid)
        valid = np.isfinite(data) & (data >= 0)
        mask = valid.astype(np.float32)
        data = np.where(valid, data, 0.0)
        # Normalize to [0, 1] using the training statistic; extremes may slightly exceed 1
        data = data / self.data_max

        # Output (T, H, W, 2): channel 0 = reflectivity, channel 1 = validity mask
        vid = np.zeros(
            (self.total_length, self.img_height, self.img_width, 2),
            dtype=np.float32,
        )
        vid[..., 0] = data
        vid[..., 1] = mask

        return torch.from_numpy(vid)
