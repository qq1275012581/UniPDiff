import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import argparse
from tqdm import tqdm
import random
import datetime

from arch import ResCast
import torch
from data.mean_std import NAME_MAPPING_DICT, get_all_vars_mean_std
from data.cal_grid_metrics import era5_lat_weighted_rmse
from torchvision import transforms
from inference_pangu import pengu_inference

model_data_dict = {
    'pangu': '/home-ssd/Users/gm_intern/liguowen/UniPDiff/output_data/',
}

era5_surface_vars = ['t2m', 'msl', 'u10', 'v10']
era5_upper_vars = ['t', 'z', 'u', 'v', 'q']
levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

lat = np.arange(90, -90-1.5, -1.5)
lon = np.arange(0, 360, 1.5)
gpm_data_dir = "/home-ssd/Users/gm_intern/liguowen/data/ERA5/1p5deg/npy_data/gpm_tp_24hr/"

def plot_comparison(pred, target, time_str, save_dir, model_name, data_feature, pred_day=None):
    """
    绘制预测值与真值的对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 4), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    
    # 定义降水常用的颜色分段和 Colormap (从浅蓝到深紫)
    levels = [0, 0.1, 1, 2, 5, 10, 20, 50, 100]
    colors = ['#ffffff', '#87CEEB', '#00BFFF', '#1E90FF', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # 设置绘图范围 [左, 右, 下, 上]
    img_extent = [0, 360, -90, 90]

    # 绘制预测图
    im0 = axes[0].imshow(pred.squeeze(), cmap=cmap, norm=norm, extent=img_extent, origin='upper')
    axes[0].add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333', zorder=2)
    axes[0].set_title(f'{model_name} Prediction (24h TP)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    axes[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    plt.colorbar(im0, ax=axes[0], label='Precipitation (mm)', shrink=0.6)

    # 绘制真图 (GPM)
    im1 = axes[1].imshow(target.squeeze(), cmap=cmap, norm=norm, extent=img_extent, origin='upper')
    axes[1].add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333', zorder=2)
    axes[1].set_title('GPM Ground Truth (24h TP)', fontsize=14)
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    axes[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    plt.colorbar(im1, ax=axes[1], label='Precipitation (mm)', shrink=0.6)

    plt.suptitle(f'Precipitation Comparison - {time_str} lead time at Days {pred_day}', fontsize=16)
    
    # 保存图片
    save_path = os.path.join(save_dir, 'fig', model_name, data_feature, f'vis_comparison_rescast_{time_str}_pred_day{pred_day}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")

def plot_pred_list(pred_list, time_str, save_dir, model_name, data_feature, denoise_step_list, idx_day=0):
    """
    绘制一系列降水预测图，每行 3 张，已修复 0-360 经度对齐问题。
    :param pred_list: 包含多个 (1, H, W) 或 (H, W) 数组的列表
    :param time_str: 时间字符串
    :param save_dir: 保存路径
    :param model_name: 模型名称
    :param denoise_step_list: 降噪步骤列表
    """
    num_plots = len(pred_list)
    cols = 3
    rows = (num_plots + cols - 1) // cols  # 向上取整计算行数

    # 1. 创建画布，必须指定 projection 为地理投影
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows),
                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    # fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows),
    #                          subplot_kw={'projection': ccrs.Orthographic(central_longitude=10, central_latitude=20)})
    
    # 统一子图数组格式
    if num_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 降水配色方案
    levels = [0, 0.1, 1, 2, 5, 10, 20, 50, 100]
    colors = ['#ffffff', '#87CEEB', '#00BFFF', '#1E90FF', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im = None
    for i in range(num_plots):
        ax = axes[i]
        
        # 获取数据并去除多余维度
        # print(pred_list[i][idx_day].shape)
        pred = np.squeeze(pred_list[i][idx_day])
        H, W = pred.shape
        
        # --- 核心修改：动态生成经纬度并添加循环点以修复错位 ---
        # 假设数据是全球等分辨率排列，经度从 0 开始
        lons = np.linspace(0, 360, W, endpoint=False)
        lats = np.linspace(90, -90, H)
        
        # add_cyclic_point 会在数据末尾增加一列 360° 的数据，消除 0/360 处的缝隙
        cyclic_data, cyclic_lons = add_cyclic_point(pred, coord=lons)
        
        # 2. 绘图：改用 pcolormesh，它对地理坐标支持比 imshow 更精确
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333', zorder=2)

        im = ax.pcolormesh(cyclic_lons, lats, cyclic_data, 
                           cmap=cmap, norm=norm, 
                           shading='auto', transform=ccrs.PlateCarree())
        
        # 3. 设置样式
        if i < len(denoise_step_list):
            step = denoise_step_list[i]
            ax.set_title(f'{time_str}_{step}', fontsize=16)
        else:
            ax.set_title(f'{time_str} (ground truth)', fontsize=16)
        
        # 设置经纬度刻度
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
        # ax.set_xticks(range(0, 361, 60), crs=ccrs.PlateCarree())
        ax.set_yticks([-90, -45, 0, 45, 90], crs=ccrs.PlateCarree())
        
        if i % cols == 0:
            ax.set_ylabel('Latitude (°N)', fontsize=12)
        if i >= (rows - 1) * cols:
            ax.set_xlabel('Longitude (°E)', fontsize=12)

    # 4. 隐藏多余的子图
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    # 5. 添加全局颜色条
    fig.subplots_adjust(right=0.92, wspace=0.1, hspace=0.25)
    cbar_ax = fig.add_axes([0.94, 0.2, 0.015, 0.6])
    fig.colorbar(im, cax=cbar_ax, label='Precipitation (mm)')

    # 6. 保存
    save_path = os.path.join(save_dir, 'fig', model_name, data_feature, f'batch_vis_{time_str}_{idx_day}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Batch visualization saved to {save_path}")

def get_args():
    parser = argparse.ArgumentParser('Evaluation Model', add_help=False)
    parser.add_argument('--ckpt_pth', type=str, default='/path/to/your/model.pth', help='model path')
    parser.add_argument('--output_dir', type=str, default='./output', help='output path')
    parser.add_argument('--model_name', type=str, default='graphcast', help='model name: graphcast or rescast')
    parser.add_argument('--start_time', type=str, default='2022010200', help='start time for evaluation')
    parser.add_argument('--end_time', type=str, default='2022010200', help='end time for evaluation')
    parser.add_argument('--idx_of_day', type=int, nargs='+', default=[0, 1, 2, 3], 
                        help='idx list of time in day')
    
    parser.add_argument('--sample_steps', type=int, default=100, help='number of the sample steps')
    parser.add_argument('--save_tp', action='store_true', help='whether to save only tp24h predictions')
    parser.add_argument('--visualize', action='store_true', help='whether to visualize predictions vs ground truth')

    # distributed parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()
    return args

def get_time_str_list(start_time_str, end_time_str, hours=24):
    start_time = datetime.datetime.strptime(start_time_str, '%Y%m%d%H')
    end_time = datetime.datetime.strptime(end_time_str, '%Y%m%d%H')
    time_str_list = []
    current_time = start_time
    while current_time <= end_time:
        time_str_list.append(current_time.strftime('%Y%m%d%H'))
        current_time += datetime.timedelta(hours=hours)
    return time_str_list

def get_var_names(surf_vars, atmo_vars, levels):
    var_names = surf_vars.copy()
    for var in atmo_vars:
        for level in levels:
            var_names.append(f"{var}_{level}")
    return var_names

def get_rescast_model(args, var_names, device=None):

    select_model = ResCast
    model_config = {
        "var_names": var_names,
        "embed_dim": 384,
        "T_in": len(args.idx_of_day)
    }

    model = select_model(**model_config)
    model.to(device)
    return model 


def get_model_pred_data(args, var_names, time_str, vars_transform, device=None):

    if args.model_name == 'pangu':
        surface_data, upper_data = pengu_inference(time_str)
        input_data = []
        for var in var_names:
            data = np.load(os.path.join(model_data_dict[args.model_name], var, f"{time_str}.npy"))
            input_data.append(data)
        input_data = np.stack(input_data, axis=1)

        input_data = torch.Tensor(input_data).to(device)[:, :, 1:, :]

        input_data = vars_transform(input_data).unsqueeze(0)

        T = input_data.shape[1] // 4

    else:
        raise ValueError(f"Unknown backbone name: {args.model_name}")

    gpm_tp24h = []
    for i in range(T):
        time_pd = datetime.datetime.strptime(time_str, '%Y%m%d%H') + datetime.timedelta(days=i)
        time_str_i = time_pd.strftime('%Y%m%d%H')
        gpm_tp24h.append(np.load(os.path.join(gpm_data_dir, f'{time_str_i}.npy'))[np.newaxis, :, :])
    gpm_tp24h = np.concatenate(gpm_tp24h, axis=0)[:, 1:][np.newaxis, :, np.newaxis, :]

    return input_data, gpm_tp24h

def reshape_data_from_idx(data, args):

    data_list = []
    for i in range(0, data.shape[1], 4):
        idx_list = []
        for idx in args.idx_of_day:
            idx_list.append(idx+i)
        data_list.append(data[:, idx_list])
    return data_list


def lat_weighted_rmse(predict: np.ndarray, truth: np.ndarray, lat: np.ndarray) -> float:
    """
    针对 (1, 1, H, W) 输入计算纬度加权 RMSE，直接返回单个浮点数值。
    
    Parameters:
    -----------
    predict/truth : np.ndarray
        形状为 (B, C, H, W) 的数据，当前场景为 (1, 1, H, W)。
    lat : np.ndarray
        纬度数组，形状为 (H,)。
    """
    # 1. 确保输入为 4 维
    assert predict.ndim == 4, "Input must be (B, C, H, W)"
    
    # 2. 计算纬度权重并归一化 (参考你提供的逻辑)
    # cos(lat) 越大，权重越高；除以 mean() 是为了保持整体数值量纲
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H,)
    
    # 3. 调整权重形状以广播至 (B, C, H, W) -> (1, 1, H, 1)
    w_lat = w_lat[np.newaxis, np.newaxis, :, np.newaxis]
    
    # 4. 计算加权平方误差
    # 根据参考代码逻辑: np.mean(w_lat * (error^2))
    # 这会自动对所有维度 (B, C, H, W) 取平均
    diff_sq = (predict - truth) ** 2
    weighted_rmse = np.sqrt(np.mean(w_lat * diff_sq))
    
    return float(weighted_rmse)

def set_seed(seed=42):
    # 1. Python 自带随机库
    random.seed(seed)
    # 2. Numpy 随机库
    np.random.seed(seed)
    # 3. PyTorch 随机种子 (CPU)
    torch.manual_seed(seed)
    # 4. PyTorch 随机种子 (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 时使用
    
    # 5. 确定性操作（这会让生成速度略微变慢，但能保证完全一致）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")

if __name__ == '__main__':
    args = get_args()
    set_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    time_strs_list = get_time_str_list(args.start_time, args.end_time, 24)

    clim = np.load("/home-ssd/Users/gm_intern/liguowen/data/ERA5/1p5deg/npy_data/gpm_tp_24hr_clim.npy")[:, 1:]

    var_names = get_var_names(era5_surface_vars, era5_upper_vars, levels)
    mean, std = get_all_vars_mean_std(var_names)
    vars_transform = transforms.Normalize(mean=mean, std=std)

    mean, std = get_all_vars_mean_std(["gpm_tp_24hr"])
    mean_denorm, std_denorm = -mean / std, 1 / std
    transform_denorm = transforms.Normalize(mean=mean_denorm, std=std_denorm)

    model = get_rescast_model(args, var_names, device='cuda')

    if args.ckpt_pth is not None:
        state_dict = torch.load(args.ckpt_pth, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict['model'], strict=True)

    i_time  = 0
    for time_str in tqdm(time_strs_list):

        input_data, gpm_tp24h = get_model_pred_data(args, var_names, time_str, vars_transform, device='cuda')

        input_data_list = reshape_data_from_idx(input_data, args)
        
        input_data = torch.cat(input_data_list, axis=0)
        
        output_tp, denoise_img_list = model.evaluate_ddim(input_data, steps=args.sample_steps)

        output_tp = transform_denorm(output_tp)
        output_tp = torch.clamp((torch.exp(output_tp) - 1.0) * 1e-2, min=0.0)  # Inverse of log1p
        output_tp = output_tp.cpu().numpy()

        output_tp = output_tp[np.newaxis, :]
        print(output_tp.shape, gpm_tp24h.shape, clim.shape)
        T = gpm_tp24h.shape[1]
        if clim.ndim == 3:
            clim = np.repeat(clim[np.newaxis, ...], repeats=T, axis=0)

        rmse_metrics = era5_lat_weighted_rmse(output_tp, gpm_tp24h, 
                                    variable_names=["gpm_tp"], predict_avg=clim, truth_avg=clim, lat=lat[1:])

        rmse = np.sqrt(((output_tp - gpm_tp24h) ** 2).mean(axis=(0, 2, 3, 4)))

        if args.save_tp:
            save_path = os.path.join(args.output_dir, 'data', args.model_name, 
                                    f'2idx_{args.sample_steps}step', f'{time_str}.npy')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, output_tp)


        if args.visualize and i_time % 10 == 0 and not args.ensemble_forecast:
            data_feature = f'2idx_{args.sample_steps}step'
            vis_idx_list = [1, 3, 5, 7] # pred_day
            for i in vis_idx_list:
                plot_comparison(output_tp[:, i-1], gpm_tp24h[:, i-1], time_str, args.output_dir, 
                                args.model_name, data_feature, pred_day=i)
            for i, img in enumerate(denoise_img_list):
                img = transform_denorm(img)
                img = torch.clamp((torch.exp(img) - 1.0) * 1e-2, min=0.0)  # Inverse of log1p
                img = img.cpu().numpy()
                denoise_img_list[i] = img

            idx_day = 2
            # print(gpm_tp24h[0, idx_day].shape)
            denoise_img_list.append(gpm_tp24h[0])
            denoise_step_list = np.arange(1000, -1, -100)
            plot_pred_list(denoise_img_list, time_str, args.output_dir, args.model_name, data_feature, denoise_step_list, idx_day=idx_day)

        i_time += 1



        

        
