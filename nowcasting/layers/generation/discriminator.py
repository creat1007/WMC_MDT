import torch
import torch.nn as nn
import torch.nn.functional as F
from nowcasting.layers.utils import spectral_norm


class Temporal_Discriminator(nn.Module):
    """Spatiotemporal discriminator (NowcastNet / DGMR style).

    Takes the full sequence (B, T, H, W) -- input frames concatenated with future frames --
    and judges whether its spatiotemporal evolution "looks like real radar". 3D convolutions
    capture spatiotemporal structure, then the time axis is pooled away and 2D convolutions
    continue downsampling. Spectral norm stabilizes training; the output is a score map
    (used with a hinge loss).

    The first layer downsamples aggressively (stride=(2,4,4)) to keep memory in check.
    """

    def __init__(self, in_frames, base_c=32):
        super().__init__()
        sn = spectral_norm
        # 3D spatiotemporal convolutions
        self.c3d_1 = sn(nn.Conv3d(1, base_c, kernel_size=(4, 9, 9),
                                  stride=(2, 4, 4), padding=(1, 4, 4)))
        self.c3d_2 = sn(nn.Conv3d(base_c, base_c * 2, kernel_size=(4, 5, 5),
                                  stride=(2, 2, 2), padding=(1, 2, 2)))
        # Pool away the time axis, then continue downsampling in 2D
        self.c2d_1 = sn(nn.Conv2d(base_c * 2, base_c * 4, kernel_size=3, stride=2, padding=1))
        self.c2d_2 = sn(nn.Conv2d(base_c * 4, base_c * 8, kernel_size=3, stride=2, padding=1))
        self.c2d_3 = sn(nn.Conv2d(base_c * 8, base_c * 8, kernel_size=3, stride=1, padding=1))
        self.out = sn(nn.Conv2d(base_c * 8, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        # x: (B, T, H, W) → (B, 1, T, H, W)
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.c3d_1(x), 0.2)
        x = F.leaky_relu(self.c3d_2(x), 0.2)
        x = x.mean(dim=2)                      # collapse time -> (B, C, H', W')
        x = F.leaky_relu(self.c2d_1(x), 0.2)
        x = F.leaky_relu(self.c2d_2(x), 0.2)
        x = F.leaky_relu(self.c2d_3(x), 0.2)
        return self.out(x)                     # (B, 1, h, w) score map


# -- Losses -----------------------------------------------------------------

def hinge_loss_d(real_score, fake_score):
    """Discriminator hinge loss: push real samples above +1 and fakes below -1."""
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()


def hinge_loss_g(fake_score):
    """Generator adversarial loss: make the discriminator score fakes as real."""
    return -fake_score.mean()


def pool_regularization(gen, real, kernel=8):
    """Pooling regularization (NowcastNet): keeps generated and real fields consistent at
    coarse scale (area means), preventing the discriminator from pushing the generator to
    fabricate unrealistic strong echoes. gen/real: (B, T, H, W)."""
    b, t, h, w = gen.shape
    g = F.avg_pool2d(gen.reshape(b * t, 1, h, w), kernel)
    r = F.avg_pool2d(real.reshape(b * t, 1, h, w), kernel)
    return F.l1_loss(g, r)
