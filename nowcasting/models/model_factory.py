import os
import torch
from nowcasting.models import nowcastnet

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        networks_map = {'NowcastNet': nowcastnet.Net}
        
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(configs).to(configs.device)
            self.test_load()
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

    def test_load(self):
        if not self.configs.pretrained_model or not os.path.exists(self.configs.pretrained_model):
            print(f"[ERROR] 找不到 checkpoint: {self.configs.pretrained_model} → 使用随机权重！", flush=True)
            return
        
        try:
            ckpt = torch.load(self.configs.pretrained_model, map_location=self.configs.device, weights_only=False)
            # checkpoint 可能是完整字典或纯 state_dict
            state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self.network.load_state_dict(state_dict, strict=False)
            n_model = len(self.network.state_dict())
            n_loaded = n_model - len(missing)
            print(f"[INFO] 加载模型: {self.configs.pretrained_model}", flush=True)
            print(f"[INFO] 权重匹配: {n_loaded}/{n_model} 个张量成功加载 "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
            if n_loaded == 0:
                print("[ERROR] 没有任何权重被加载！checkpoint 的 key 与模型不匹配 → 实际是随机权重！", flush=True)
            elif len(missing) > 0:
                print(f"[WARNING] 有 {len(missing)} 个张量未加载(用随机值)，例如: {missing[:5]}", flush=True)
        except Exception as e:
            print(f"[WARNING] 模型加载失败: {e}，使用随机初始化", flush=True)

    def test(self, frames):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        self.network.eval()
        with torch.no_grad():
            next_frames = self.network(frames_tensor)
        return next_frames.detach().cpu().numpy()
