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
            print(f"[ERROR] Checkpoint not found: {self.configs.pretrained_model} -> using RANDOM weights!", flush=True)
            return
        
        try:
            ckpt = torch.load(self.configs.pretrained_model, map_location=self.configs.device, weights_only=False)
            # Checkpoint may be a full dict or a bare state_dict
            state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self.network.load_state_dict(state_dict, strict=False)
            n_model = len(self.network.state_dict())
            n_loaded = n_model - len(missing)
            print(f"[INFO] Loading model: {self.configs.pretrained_model}", flush=True)
            print(f"[INFO] Weights matched: {n_loaded}/{n_model} tensors loaded "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
            if n_loaded == 0:
                print("[ERROR] No weights were loaded! Checkpoint keys do not match the model -> weights are RANDOM!", flush=True)
            elif len(missing) > 0:
                print(f"[WARNING] {len(missing)} tensors not loaded (random init), e.g.: {missing[:5]}", flush=True)
        except Exception as e:
            print(f"[WARNING] Model loading failed: {e}; using random initialization", flush=True)

    def test(self, frames):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        self.network.eval()
        with torch.no_grad():
            next_frames = self.network(frames_tensor)
        return next_frames.detach().cpu().numpy()
