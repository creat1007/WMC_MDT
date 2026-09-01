import os
import json
import threading
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


def compute_data_stats(data_path, percentile=99.9, cache_file=None):
    """
    扫描所有训练NC文件，计算指定百分位数作为归一化基准值 data_max。
    对每个文件独立计算 percentile，取所有文件中的最大值作为全局 data_max。

    Args:
        data_path   : NC文件目录或单个NC文件路径
        percentile  : 百分位数，默认99.9（比全局最大值更鲁棒）
        cache_file  : 缓存路径（JSON），存在则直接读取，跳过扫描

    Returns:
        data_max (float)
    """
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            stats = json.load(f)
        print(f"[Stats] 从缓存读取 data_max={stats['data_max']:.4f} (p{stats['percentile']}, {stats['n_files']}个文件)", flush=True)
        return float(stats['data_max'])

    nc_files = _collect_nc_files(data_path)
    if not nc_files:
        raise FileNotFoundError(f"未找到NC文件: {data_path}")

    print(f"[Stats] 开始扫描 {len(nc_files)} 个文件，计算 p{percentile}...", flush=True)
    per_file_pct = []

    for i, nc_path in enumerate(nc_files):
        try:
            import xarray as xr
            with xr.open_dataset(nc_path, decode_times=False) as ds:
                var_names = list(ds.data_vars)
                if not var_names:
                    continue
                data = ds[var_names[0]].values.astype(np.float32)
            # 单位为 0.1 dBZ，先转换为真实 dBZ
            data = data / 10.0
            valid = data[np.isfinite(data) & (data >= 0)]
            if valid.size > 0:
                per_file_pct.append(float(np.percentile(valid, percentile)))
        except Exception as e:
            print(f"  Warning: 跳过 {os.path.basename(nc_path)}: {e}", flush=True)

        if (i + 1) % 100 == 0 or (i + 1) == len(nc_files):
            cur_max = max(per_file_pct) if per_file_pct else 0.0
            print(f"  [{i+1}/{len(nc_files)}] 当前最大p{percentile}: {cur_max:.4f}", flush=True)

    if not per_file_pct:
        raise ValueError("无法从训练数据中计算统计量，请检查NC文件格式")

    data_max = float(np.max(per_file_pct))
    print(f"[Stats] 扫描完成: data_max={data_max:.4f} (p{percentile}，共{len(per_file_pct)}个有效文件)", flush=True)

    if cache_file:
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({'data_max': data_max, 'percentile': percentile, 'n_files': len(per_file_pct)}, f, indent=2)
        print(f"[Stats] 统计结果已缓存至: {cache_file}", flush=True)

    return data_max


def nc_to_cache_name(nc_path):
    """把 NC 文件的完整路径映射成唯一的缓存文件名（避免不同子目录重名冲突）。
    转换脚本和 Dataset 必须用同一个映射。"""
    key = os.path.abspath(nc_path).replace(os.sep, '__').lstrip('_')
    return key + '.npy'


def _collect_nc_files(path):
    """
    递归收集 path 下（或 path 本身）所有 .nc 文件，返回排序后的路径列表。
    path 可以是：
      - 单个 .nc 文件路径
      - 含 .nc 文件的文件夹（支持多级子文件夹）
      - 逗号分隔的多个目录（如华北+华南混合训练：
        "/data/radar/training_data_Pek,/data/radar/training_data_GBA"）
    """
    # 逗号分隔的多路径：分别收集后合并（华北+华南混合训练）
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
        raise ValueError(f"{path} 不是 .nc 文件")

    nc_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.nc'):
                nc_files.append(os.path.join(root, f))
    return sorted(nc_files)


class RadarTrainDataset(Dataset):
    """
    NowcastNet 训练数据集，支持多种数据组织方式：

      ① 单个大 .nc 文件（包含所有时次，shape=(T_total, H, W)）
      ② 每天一个 .nc 文件（每文件帧数任意，不要求为 240）
      ③ 按月/年组织的多级文件夹（递归扫描）
      ④ 以上混合

    对每个 .nc 文件，以 stride 为步长生成滑动窗口样本：
      sample = (nc文件路径, 起始帧索引)
    帧数 < total_length 的文件自动跳过。

    Args:
        data_path   : 单个 .nc 文件路径，或含 .nc 文件的文件夹（支持多级）
        input_length: 输入帧数（默认 20）
        total_length: 输入+预报总帧数（默认 50）
        stride      : 滑窗步长（默认 1；设为 6 可减少约 6 倍样本量）
        img_height  : 目标图像高度（默认 512）
        img_width   : 目标图像宽度（默认 512）
    """

    def __init__(self, data_path, input_length=9, total_length=29,
                 stride=1, img_height=512, img_width=512, data_max=80.0,
                 _prebuilt_samples=None, cache_dir=None):
        self.input_length = input_length
        self.total_length = total_length
        self.img_height   = img_height
        self.img_width    = img_width
        self.data_max     = data_max
        # 预处理缓存目录（每个 NC 对应一个已 resize/转 dBZ/翻转的 .npy）。
        # 提供且命中时直接 mmap 读取，跳过开 NC + cv2.resize 的瓶颈。
        self.cache_dir    = cache_dir
        # netCDF/HDF5 非线程安全：回退到 NC 读取时用锁串行化，避免并发 SIGSEGV。
        # 缓存(.npy mmap)读取不走这把锁，仍可多线程并行。
        self._nc_lock     = threading.Lock()

        # 如果外部已建好索引（rank 0 广播），直接使用，跳过磁盘扫描
        if _prebuilt_samples is not None:
            self.samples = _prebuilt_samples
            return

        nc_files = _collect_nc_files(data_path)
        if not nc_files:
            raise FileNotFoundError(f"在 {data_path} 下未找到任何 .nc 文件")

        # 构建样本索引：[(nc_path, var_name, start_frame_idx), ...]
        self.samples  = []
        skipped_files = 0

        for nc_path in nc_files:
            var_name, n_frames = self._probe_file(nc_path)
            if var_name is None or n_frames < total_length:
                skipped_files += 1
                continue
            # 把变量名一并存入索引，避免重复探测
            for start in range(0, n_frames - total_length + 1, stride):
                self.samples.append((nc_path, var_name, start))

        print(
            f"[Dataset] 扫描 {len(nc_files)} 个文件，"
            f"跳过 {skipped_files} 个（帧数不足 {total_length}），"
            f"生成 {len(self.samples)} 个样本（stride={stride}）"
        )

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _probe_file(self, nc_path):
        """
        打开 NC 文件，返回 (var_name, n_frames)。
        失败或无变量时返回 (None, 0)。
        """
        try:
            import xarray as xr
            with xr.open_dataset(nc_path, decode_times=False) as ds:
                var_names = list(ds.data_vars)
                if not var_names:
                    return None, 0
                var = var_names[0]
                values = ds[var].values
                # 兼容不同维度：取第一个维度作为时间轴
                if values.ndim < 2:
                    return None, 0
                if values.ndim == 2:
                    # 单帧文件（H, W） → 视为 1 帧
                    return var, 1
                return var, int(values.shape[0])
        except Exception as e:
            print(f"  Warning: 无法读取 {os.path.basename(nc_path)}: {e}")
            return None, 0

    def _load_frames(self, nc_path, var_name, start):
        """
        从 nc_path 读取 [start, start+total_length) 帧。
        返回 shape=(total_length, img_height, img_width) 的 float32 数组。
        """
        # ── 缓存快路径：直接 mmap 读取已预处理好的 .npy ──────────────────
        if self.cache_dir is not None:
            cpath = os.path.join(self.cache_dir, nc_to_cache_name(nc_path))
            if os.path.exists(cpath):
                arr = np.load(cpath, mmap_mode='r')  # (T_full, H, W)，已 dBZ/翻转/resize
                data = np.asarray(arr[start:start + self.total_length], dtype=np.float32)
                # 防御：缓存尺寸与目标不一致时再 resize（正常不会触发）
                if data.shape[1] != self.img_height or data.shape[2] != self.img_width:
                    data = np.stack([
                        cv2.resize(f, (self.img_width, self.img_height),
                                   interpolation=cv2.INTER_LINEAR)
                        for f in data
                    ])
                return data.astype(np.float32)
            # 未命中则回退到原始 NC 读取路径

        import xarray as xr
        # netCDF/HDF5 非线程安全，加锁串行化，避免多线程并发读取导致 SIGSEGV
        with self._nc_lock:
            with xr.open_dataset(nc_path, decode_times=False) as ds:
                da = ds[var_name]
                # 只读需要的帧，避免把整个文件加载进内存
                if da.ndim >= 3:
                    time_dim = da.dims[0]
                    raw = da.isel({time_dim: slice(start, start + self.total_length)}).values
                else:
                    raw = da.values  # 单帧文件

        # 统一为 (T, H, W)
        if raw.ndim == 2:
            raw = raw[np.newaxis, ...]

        data = raw[:self.total_length].copy()  # (T, H, W)

        # 数据单位为 0.1 dBZ（存储值 = dBZ × 10），转换为真实 dBZ
        data = data / 10.0

        # 反转 y 轴（与现有 loader 保持一致）
        data = data[:, ::-1, :]

        # 缩放到目标分辨率
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

    # ── Dataset 接口 ──────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        nc_path, var_name, start = self.samples[index]
        data = self._load_frames(nc_path, var_name, start)  # (T, H, W)

        # 有效值掩膜（缺测/负值视为无效）
        valid = np.isfinite(data) & (data >= 0)
        mask = valid.astype(np.float32)
        data = np.where(valid, data, 0.0)
        # 用训练集统计量归一化到 [0, 1]，极端值允许略超1
        data = data / self.data_max

        # 输出 (T, H, W, 2)：channel-0=雷达反射率，channel-1=有效掩膜
        vid = np.zeros(
            (self.total_length, self.img_height, self.img_width, 2),
            dtype=np.float32,
        )
        vid[..., 0] = data
        vid[..., 1] = mask

        return torch.from_numpy(vid)
