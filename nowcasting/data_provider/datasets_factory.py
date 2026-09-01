from nowcasting.data_provider import loader
from torch.utils.data import DataLoader

datasets_map = {'radar': loader}

def data_provider(configs):
    if configs.dataset_name == 'radar':
        test_input_param = {
            'image_width': configs.img_width,
            'image_height': configs.img_height,
            'input_data_type': 'float32',
            'total_length': configs.total_length,
            'data_path': configs.dataset_path,
            'type': 'test',
            'forecast_only': getattr(configs, 'forecast_only', False),
            'input_length': configs.input_length,
            'data_max': getattr(configs, 'data_max', 80.0),
        }
        test_input_handle = datasets_map[configs.dataset_name].InputHandle(test_input_param)
        test_input_handle = DataLoader(test_input_handle,
                                       batch_size=configs.batch_size,
                                       shuffle=False,
                                       num_workers=configs.cpu_worker,
                                       drop_last=True)
    else:
        raise ValueError('Name of dataset unknown %s' % configs.dataset_name)

    return test_input_handle
