# WMC_MDT — 雷达回波临近预报模型

基于 [NowcastNet](https://www.nature.com/articles/s41586-023-06184-4) 的雷达外推临近预报系统，
针对中国区域（华北 / 华南）业务化改造与训练。

**输入过去 20 帧（2 小时），预报未来 30 帧（3 小时），帧间隔 6 分钟。**

---

## 主要特性

- **物理-生成混合架构**：Evolution 网络学习光流平流演变，Generative 网络（含判别器）负责细节生成
- **对抗训练（GAN）**：时空判别器 + 池化正则，输出锐利、有强回波核的预报场，避免 L1 回归的过度平滑
- **频率平衡损失**：前景/背景分别求均值，解决雷达场 99% 是零背景导致的模型塌缩
- **多区域混合训练**：支持华北 + 华南数据混合，一个模型覆盖多区域
- **业务化就绪**：单层（组合反射率）+ 多层（12 个高度层）双流水线，支持实时定时运行

---

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

### 训练

```bash
python train.py \
    --train_data_path /path/to/data_regionA,/path/to/data_regionB \
    --data_max 55.5 \
    --gan --lambda_adv 0.03 --lambda_pool 0.5 \
    --lambda_motion 0.0003 --lambda_evo 3.0 \
    --warp_mode bilinear \
    --lr 1e-4 --batch_size 12 \
    --save_dir ./checkpoints
```

多卡：`torchrun --nproc_per_node=4 train.py ...`

**从已有模型继续训练（warm start）**：

```bash
python train.py --pretrained_model ./checkpoints/best_model.ckpt --data_max 55.5 ...
```

> ⚠️ 换数据集做 warm start 时，`--data_max` 必须与预训练一致，否则归一化尺度改变会使已训权重失效。

### 推理

```bash
python run.py \
    --forecast_only \
    --dataset_path /path/to/input_dir \
    --pretrained_model ./checkpoints/best_model.ckpt \
    --gen_frm_dir ./results \
    --input_length 20 --total_length 50 \
    --img_height 512 --img_width 512 \
    --warp_mode bilinear
```

### 数据预处理（可选，大幅加速）

把 NetCDF 预处理成 `.npy` 缓存，训练时 mmap 直读：

```bash
python preprocess_cache.py --data_path /path/to/nc_dir --cache_dir /path/to/cache
```

彩色雷达图（PNG）转 dBZ 训练数据：

```bash
python preprocess_cache.py --input_type image \
    --data_path /path/to/png_dir --cache_dir /path/to/npy \
    --pal_file /path/to/palette.txt
```

---

## 目录结构

```
├── train.py                    # 训练主程序（GAN / warm start / 多卡）
├── run.py                      # 推理
├── preprocess_cache.py         # 数据预处理（NC / 彩色图 → npy）
├── revive_flow.py              # 修复光流层权重坍缩（见下）
├── diagnose_typhoon.py         # 诊断：区分平流阶段 vs 生成阶段的结构损失
└── nowcasting/
    ├── models/nowcastnet.py            # 主模型
    ├── layers/evolution/               # Evolution 网络（光流 + 强度）
    ├── layers/generation/              # Generative 网络
    │   └── discriminator.py            # 时空判别器 + hinge loss + 池化正则
    └── data_provider/                  # 数据加载
```

---

## 训练要点

### 关键超参

| 参数 | 说明 | 建议值 |
|---|---|---|
| `--data_max` | 归一化基准（dBZ）。warm start 时必须与预训练一致 | `55.5` |
| `--lambda_adv` | 对抗损失权重。太大易崩，太小无锐化效果 | `0.03` |
| `--lambda_pool` | 池化正则，约束粗尺度水量、防止判别器逼模型乱造强回波 | `0.5` |
| `--lambda_evo` | Evolution 自监督权重 | `3.0` |
| `--lambda_motion` | 光流平滑正则。过大会压制旋转等高梯度流场 | `0.0003` |
| `--warp_mode` | **必须用 `bilinear`**（见下） | `bilinear` |

### 评估

训练中每 `--val_interval` 轮在留出验证集上计算 **CSI**（20/30/40 dBZ）。

> **注意**：GAN 训练下 CSI 可能略降而观感明显变好——CSI 惩罚"位置稍偏的锐利强核"，
> 却奖励模糊的平均场。判断模型质量应以**个例可视化为主，CSI 为辅**。

---

## 开发中发现的关键问题

记录几个排查过程中定位到的、影响很大的问题，供参考。

### 1. 模型塌缩为空白输出

纯 L1 损失下，雷达场 99% 是零背景，模型输出全零即可让 loss 降到 ~1e-5，
形成"假收敛"（loss 持续下降但 CSI 恒为 0）。

**解决**：`balanced_l1` —— 前景（≥15 dBZ）与背景分别求均值再加权相加，
前景不被海量背景稀释；并引入有效掩膜、按 dBZ 分级加权。

### 2. 预报过度平滑、强度塌缩

L1 回归倾向输出"安全的平均场"：强核被抹平、回波摊成大片弱区。

**解决**：重新引入 NowcastNet 的对抗训练（时空判别器 + hinge loss + 池化正则）。
从 L1 预训练模型 warm start 做对抗微调，比从零训 GAN 稳定得多。

### 3. 光流层权重坍缩（`revive_flow.py`）

**现象**：预报中回波几乎不移动，结构在 1 小时内被"抹匀"；
调整任何损失权重，输出的光流场都一模一样、纹丝不动。

**根因**：`warp` 使用 `mode="nearest"`，而模型预测的位移仅 ~0.02–0.2 像素/帧，
最近邻取整后位移归零 → 光流对结果无影响 → 梯度≈0 → `outc_v` 权重坍缩到 std≈1e-4
→ 输出退化为常数场 → 永远学不动（自锁死循环）。

**解决**：
1. `warp` 改用 `bilinear`（支持亚像素位移且可导）；
2. `revive_flow.py` 重新初始化坍缩的 `outc_v` 层并清空优化器状态，再 warm start。

修复后光流量级增长约 15 倍，回波恢复真实平流。

```bash
python revive_flow.py --ckpt old.ckpt --out revived.ckpt
python train.py --pretrained_model revived.ckpt --warp_mode bilinear --lr 1e-4 ...
```

---

## 已知局限

- **台风等强旋转系统**：2–3 小时后螺旋结构仍难以维持。光流量级虽已修复，
  但单一 U-Net 难以学出维持涡旋所需的连贯旋转场。1 小时内可参考。
- **对流新生**：纯雷达外推无法预报"凭空生成"的新对流，这是该类方法的固有局限。
- **移速偏慢**：系统加速/转向在过去帧中无信息，外推模型倾向低估。

后续方向：融合数值模式（NWP）风场引导 evolution 网络。

---

## 致谢

- 模型架构基于 [NowcastNet](https://github.com/thuml/NowcastNet)（Zhang et al., *Nature* 2023）

## License

见 [LICENSE](LICENSE)。
