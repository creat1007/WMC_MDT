import os
import datetime
import cv2
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def test_pytorch_loader(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.makedirs(res_path, exist_ok=True)

    # 生成完整的经纬度坐标数组（北→南方向，对应图像行方向）
    lon_min = getattr(configs, 'lon_min', 111.61)
    lon_max = getattr(configs, 'lon_max', 116.72)
    lat_min = getattr(configs, 'lat_min', 19.76)
    lat_max = getattr(configs, 'lat_max', 24.87)
    full_lon = np.linspace(lon_min, lon_max, configs.img_width)
    full_lat = np.linspace(lat_max, lat_min, configs.img_height)  # 北→南

    for batch_id, test_ims in enumerate(test_input_handle):
        test_ims = test_ims['radar_frames'].numpy()
        img_gen = model.test(test_ims)

        forecast_only = getattr(configs, 'forecast_only', False)
        output_length = configs.total_length - configs.input_length

        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.makedirs(path, exist_ok=True)

            actual_length = test_ims.shape[1]

            if configs.case_type == 'normal':
                slice_end = min(actual_length, configs.total_length)
                crop_start = 256 - 192
                crop_end = 256 + 192
                test_ims_plot = test_ims[0][:slice_end, crop_start:crop_end, crop_start:crop_end]
                forecast_length = img_gen.shape[1]
                slice_end_forecast = min(forecast_length, output_length)
                img_gen_plot = img_gen[0][:slice_end_forecast, crop_start:crop_end, crop_start:crop_end]
                lat_out = full_lat[crop_start:crop_end]
                lon_out = full_lon[crop_start:crop_end]
            else:
                # 对于large case，使用全尺寸
                slice_end = min(actual_length, configs.total_length)
                test_ims_plot = test_ims[0][:slice_end]
                forecast_length = img_gen.shape[1]
                slice_end_forecast = min(forecast_length, output_length)
                img_gen_plot = img_gen[0][:slice_end_forecast]
                lat_out = full_lat
                lon_out = full_lon

            data_max = getattr(configs, 'data_max', 80.0)

            if forecast_only:
                input_length = min(configs.input_length, test_ims_plot.shape[0])

                input_data = test_ims_plot[:input_length]
                if input_data.ndim == 4:  # 如果有通道维度
                    input_data = input_data[..., 0]  # 取数据通道

                forecast_data = img_gen_plot
                if forecast_data.ndim == 4:
                    forecast_data = forecast_data.squeeze()

                # 反归一化到物理量
                save_as_netcdf_simple(input_data * data_max, os.path.join(path, 'input.nc'), lat_out, lon_out)
                save_as_netcdf_simple(forecast_data * data_max, os.path.join(path, 'forecast.nc'), lat_out, lon_out)

                if configs.num_save_samples <= 5:  # 只保存少量PNG
                    save_pngs(input_data, os.path.join(path, 'input'), 'input', data_max)
                    save_pngs(forecast_data, os.path.join(path, 'forecast'), 'forecast', data_max)

            else:
                gt_data = test_ims_plot[..., 0] if test_ims_plot.ndim == 4 else test_ims_plot
                pred_data = img_gen_plot.squeeze()

                # 反归一化到物理量
                save_as_netcdf_simple(gt_data * data_max, os.path.join(path, 'ground_truth.nc'), lat_out, lon_out)
                save_as_netcdf_simple(pred_data * data_max, os.path.join(path, 'prediction.nc'), lat_out, lon_out)

    print('finished!')


def save_as_netcdf_simple(data, filename, lat_coords=None, lon_coords=None):
    try:
        if data.ndim == 4:
            data = data.squeeze()

        if data.ndim != 3:
            raise ValueError(f"需要3维: {data.ndim}")

        t, h, w = data.shape

        # 翻转纬度方向：图像行0为北，翻转后行0为南，lat坐标随之由南到北递增
        data = data[:, ::-1, :]

        if lat_coords is not None:
            lat_vals = lat_coords[::-1]   # 原北→南，翻转后南→北（递增）
        else:
            lat_vals = np.arange(h)

        lon_vals = lon_coords if lon_coords is not None else np.arange(w)

        ds = xr.Dataset(
            {'radar': (['time', 'lat', 'lon'], data.astype(np.float32))},
            coords={
                'time': np.arange(t),
                'lat': lat_vals,
                'lon': lon_vals,
            }
        )
        ds['lat'].attrs.update({'units': 'degrees_north', 'long_name': 'latitude'})
        ds['lon'].attrs.update({'units': 'degrees_east',  'long_name': 'longitude'})

        ds.to_netcdf(filename)
        print(f"保存NC: {filename}")

    except:
        data = data[:, ::-1, :] if data.ndim == 3 else data
        np.save(filename.replace('.nc', '.npy'), data)
        print(f"保存NPY: {filename.replace('.nc', '.npy')}")

def save_pngs(data, base_path, prefix, data_max=80.0):
    """保存PNG图片（data为归一化值，乘以data_max还原物理量）"""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    for i in range(data.shape[0]):
        img = data[i] * data_max
        img = np.clip(img, 0, 65535).astype(np.uint16)

        filename = f"{base_path}_{i+1:03d}.png"
        cv2.imwrite(filename, img)
