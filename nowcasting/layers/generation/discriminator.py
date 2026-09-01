import torch
import torch.nn as nn
import torch.nn.functional as F
from nowcasting.layers.utils import spectral_norm


class Temporal_Discriminator(nn.Module):
    """时空判别器（NowcastNet/DGMR 风格）。

    输入整段序列 (B, T, H, W)（输入帧 + 未来帧拼接），
    判别其时空演变是否「像真实雷达」。用 3D 卷积抓时空结构，再压掉时间维转 2D
    继续下采样，spectral_norm 稳定训练，输出一张打分图（配 hinge loss）。

    首层用较大空间步长(stride=(2,4,4))做激进下采样，控制显存。
    """

    def __init__(self, in_frames, base_c=32):
        super().__init__()
        sn = spectral_norm
        # 3D 时空卷积
        self.c3d_1 = sn(nn.Conv3d(1, base_c, kernel_size=(4, 9, 9),
                                  stride=(2, 4, 4), padding=(1, 4, 4)))
        self.c3d_2 = sn(nn.Conv3d(base_c, base_c * 2, kernel_size=(4, 5, 5),
                                  stride=(2, 2, 2), padding=(1, 2, 2)))
        # 压掉时间维后转 2D 继续下采样
        self.c2d_1 = sn(nn.Conv2d(base_c * 2, base_c * 4, kernel_size=3, stride=2, padding=1))
        self.c2d_2 = sn(nn.Conv2d(base_c * 4, base_c * 8, kernel_size=3, stride=2, padding=1))
        self.c2d_3 = sn(nn.Conv2d(base_c * 8, base_c * 8, kernel_size=3, stride=1, padding=1))
        self.out = sn(nn.Conv2d(base_c * 8, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        # x: (B, T, H, W) → (B, 1, T, H, W)
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.c3d_1(x), 0.2)
        x = F.leaky_relu(self.c3d_2(x), 0.2)
        x = x.mean(dim=2)                      # 压掉时间维 → (B, C, H', W')
        x = F.leaky_relu(self.c2d_1(x), 0.2)
        x = F.leaky_relu(self.c2d_2(x), 0.2)
        x = F.leaky_relu(self.c2d_3(x), 0.2)
        return self.out(x)                     # (B, 1, h, w) 打分图


# ── 损失函数 ────────────────────────────────────────────────────────────────

def hinge_loss_d(real_score, fake_score):
    """判别器 hinge 损失：真样本推向 >1，假样本推向 <-1。"""
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()


def hinge_loss_g(fake_score):
    """生成器对抗损失：让判别器认为假样本是真（分数越高越好）。"""
    return -fake_score.mean()


def pool_regularization(gen, real, kernel=8):
    """池化正则（NowcastNet）：约束生成场与真实场在粗尺度(区域平均)上一致，
    防止判别器逼生成器凭空造出不合理的强回波。gen/real: (B, T, H, W)。"""
    b, t, h, w = gen.shape
    g = F.avg_pool2d(gen.reshape(b * t, 1, h, w), kernel)
    r = F.avg_pool2d(real.reshape(b * t, 1, h, w), kernel)
    return F.l1_loss(g, r)
