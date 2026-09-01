import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nowcasting.layers.utils import warp, make_grid
from nowcasting.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
from nowcasting.layers.evolution.evolution_network import Evolution_Network
from nowcasting.layers.generation.noise_projector import Noise_Projector

class Net(nn.Module):
    def __init__(self, configs):
        super(Net, self).__init__()
        self.configs = configs
        self.pred_length = self.configs.total_length - self.configs.input_length

        self.evo_net = Evolution_Network(self.configs.input_length, self.pred_length, base_c=32)
        self.gen_enc = Generative_Encoder(self.configs.total_length, base_c=self.configs.ngf)
        self.gen_dec = Generative_Decoder(self.configs)
        self.proj = Noise_Projector(self.configs.ngf, configs)

        sample_tensor = torch.zeros(1, 1, self.configs.img_height, self.configs.img_width)
        self.register_buffer('grid', make_grid(sample_tensor))

    def forward(self, all_frames, return_evo=False):
        all_frames = all_frames[:, :, :, :, :1]

        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # Input Frames
        input_frames = frames[:, :self.configs.input_length]
        input_frames = input_frames.reshape(batch, self.configs.input_length, height, width)

        # Evolution Network
        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)
        series = []
        last_frames = all_frames[:, (self.configs.input_length - 1):self.configs.input_length, :, :, 0]
        grid = self.grid.repeat(batch, 1, 1, 1)
        # Warp interpolation mode (critical):
        #   The original "nearest" rounds sub-pixel displacement to 0. Since the predicted
        #   flow is only ~0.02-0.2 px/frame, warp effectively did nothing (it just copied
        #   the previous frame), so a typhoon relied solely on the repeatedly accumulated
        #   intensity term -- its structure got smeared out within an hour.
        #   Worse, rounding is non-differentiable, so no gradient reached the flow branch
        #   and outc_v weights collapsed (std 1e-4), leaving the flow unable to ever learn:
        #     nearest -> sub-pixel motion zeroed -> flow has no effect -> grad ~= 0 -> collapse
        #   "bilinear" is sub-pixel capable and differentiable, breaking that dead loop.
        warp_mode = getattr(self.configs, 'warp_mode', 'bilinear')
        for i in range(self.pred_length):
            last_frames = warp(last_frames, motion_[:, i], grid, mode=warp_mode, padding_mode="border")
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)
        # Data is already normalized to [0,1] in the loader; no extra scaling needed

        # Generative Network
        evo_feature = self.gen_enc(torch.cat([input_frames, evo_result], dim=1))

        noise = torch.randn(batch, self.configs.ngf, height // 32, width // 32, device=all_frames.device)
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_result)

        if return_evo:
            # evo_result: (B, pred_length, H, W) pure advection + intensity evolution (normalized)
            # motion_   : (B, pred_length, 2, H, W) optical-flow field, used for smoothness reg.
            return gen_result.unsqueeze(-1), evo_result, motion_
        return gen_result.unsqueeze(-1)