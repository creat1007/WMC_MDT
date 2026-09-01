import os
import shutil
import argparse
import torch
from nowcasting.data_provider import datasets_factory
from nowcasting.models.model_factory import Model
import nowcasting.evaluator as evaluator

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='NowcastNet')

parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--worker', type=int, default=0)
parser.add_argument('--cpu_worker', type=int, default=0)
parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--input_length', type=int, default=20)
parser.add_argument('--total_length', type=int, default=50)
parser.add_argument('--img_height', type=int, default=512)
parser.add_argument('--img_width', type=int, default=512)
parser.add_argument('--img_ch', type=int, default=2)
parser.add_argument('--case_type', type=str, default='normal')
parser.add_argument('--model_name', type=str, default='NowcastNet')
parser.add_argument('--gen_frm_dir', type=str, default='./results/nowcasting')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--warp_mode', type=str, default='bilinear', choices=['bilinear', 'nearest'],
                    help='evolution 迭代 warp 插值模式，必须与训练时一致')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--forecast_only', action='store_true')
parser.add_argument('--lon_min', type=float, default=111.61)
parser.add_argument('--lon_max', type=float, default=116.72)
parser.add_argument('--lat_min', type=float, default=19.76)
parser.add_argument('--lat_max', type=float, default=24.87)

args = parser.parse_args()

args.worker = 0
args.cpu_worker = 0

args.evo_ic = args.total_length - args.input_length
args.gen_oc = args.total_length - args.input_length
args.ic_feature = args.ngf * 10

# 从checkpoint读取data_max，保证推断归一化与训练一致
args.data_max = 80.0  # 默认值，会被checkpoint覆盖
if args.pretrained_model and os.path.exists(args.pretrained_model):
    try:
        _ckpt = torch.load(args.pretrained_model, map_location='cpu', weights_only=False)
        if isinstance(_ckpt, dict) and 'data_max' in _ckpt and _ckpt['data_max'] is not None:
            args.data_max = float(_ckpt['data_max'])
            print(f"[INFO] 从checkpoint读取 data_max={args.data_max:.4f}", flush=True)
        del _ckpt
    except Exception as e:
        print(f"[WARNING] 无法从checkpoint读取data_max，使用默认值{args.data_max}: {e}", flush=True)

def safe_makedirs(path):
    if not path.startswith('./') and not path.startswith('/'):
        path = './' + path
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path

def test_wrapper_pytorch_loader(model):
    batch_size_test = args.batch_size
    test_input_handle = datasets_factory.data_provider(args)
    args.batch_size = batch_size_test
    evaluator.test_pytorch_loader(model, test_input_handle, args, 'test_result')

args.gen_frm_dir = safe_makedirs(args.gen_frm_dir)

model = Model(args)
test_wrapper_pytorch_loader(model)

