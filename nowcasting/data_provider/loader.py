import numpy as np
import os
import torch
from torch.utils.data import Dataset
import cv2

class InputHandle(Dataset):
    def __init__(self, input_param):
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.img_width = input_param['image_width']
        self.img_height = input_param['image_height']
        self.data_path = input_param['data_path']
        
        self.forecast_only = input_param.get('forecast_only', False)
        self.input_length = input_param.get('input_length', 9)
        self.data_max = input_param.get('data_max', 80.0)

        self.case_list = []
        
        if os.path.isfile(self.data_path) and self.data_path.endswith('.nc'):
            self.case_list.append(self.data_path)
        elif os.path.isdir(self.data_path):
            items = sorted(os.listdir(self.data_path))
            for item in items:
                item_path = os.path.join(self.data_path, item)
                
                if os.path.isfile(item_path) and item_path.endswith('.nc'):
                    self.case_list.append(item_path)
                elif os.path.isdir(item_path):
                    png_files = [f for f in os.listdir(item_path) if f.endswith('.png')]
                    if len(png_files) >= self.input_length:
                        case = []
                        for i in range(self.input_length):
                            img_path = os.path.join(item_path, f"{item}-{i:02d}.png")
                            if os.path.exists(img_path):
                                case.append(img_path)
                        if len(case) == self.input_length:
                            self.case_list.append(case)

    def load_nc(self, nc_path):
        try:
            import xarray as xr
            
            ds = xr.open_dataset(nc_path, decode_times=False)
            var_names = list(ds.data_vars)
            if not var_names:
                return np.zeros((self.input_length, self.img_height, self.img_width))
            
            data = ds[var_names[0]].values
            
            # Flip y axis
            if data.ndim == 3:
                data = data[:, ::-1, :]
            elif data.ndim == 2:
                data = data[::-1, :]
                data = data[np.newaxis, ...]
            elif data.ndim == 4:
                if data.shape[-1] == 1:
                    data = data[..., 0]
                    data = data[:, ::-1, :]
                else:
                    data = data.squeeze()
                    if data.ndim == 3:
                        data = data[:, ::-1, :]
            
            if data.ndim != 3:
                return np.zeros((self.input_length, self.img_height, self.img_width))
            
            # Stored unit is 0.1 dBZ (value = dBZ x 10); convert to true dBZ
            data = data / 10.0

            if data.shape[0] < self.input_length:
                last_frame = data[-1:]
                padding = np.tile(last_frame, (self.input_length - data.shape[0], 1, 1))
                data = np.concatenate([data, padding], axis=0)
            
            if data.shape[1] != self.img_height or data.shape[2] != self.img_width:
                new_data = []
                for i in range(data.shape[0]):
                    resized = cv2.resize(data[i], (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
                    new_data.append(resized)
                data = np.array(new_data)
            
            return data.astype(self.input_data_type)
            
        except:
            return np.zeros((self.input_length, self.img_height, self.img_width))

    def load_imgs(self, img_paths):
        data = []
        for img_path in img_paths:
            img = cv2.imread(img_path, 2)
            data.append(np.expand_dims(img, axis=0))
        
        # Match training / NetCDF path: only /10 to dBZ (drop the legacy -3.0 offset)
        data = np.concatenate(data, axis=0).astype(self.input_data_type) / 10.0
        
        if data.shape[1] != self.img_height or data.shape[2] != self.img_width:
            new_data = []
            for i in range(data.shape[0]):
                resized = cv2.resize(data[i], (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
                new_data.append(resized)
            data = np.array(new_data)
        
        return data

    def __getitem__(self, index):
        item = self.case_list[index]
        
        if isinstance(item, str) and item.endswith('.nc'):
            data = self.load_nc(item)
        else:
            data = self.load_imgs(item)
        
        if data.shape[0] > self.input_length:
            data = data[:self.input_length]
        
        mask = np.ones_like(data)
        mask[data < 0] = 0
        data[data < 0] = 0
        # Match training: normalize to [0, 1] with data_max
        data = data / self.data_max
        
        vid = np.zeros((self.input_length, self.img_height, self.img_width, 2))
        vid[..., 0] = data
        vid[..., 1] = mask
        
        return {'radar_frames': vid}

    def __len__(self):
        return len(self.case_list)
