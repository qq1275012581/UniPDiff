from einops import rearrange
from matplotlib.pylab import f
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from mean_std import get_all_vars_mean_std
except ImportError:
    from .mean_std import get_all_vars_mean_std

TEMPORAL_KEYS = ('daily', 'monthly', 'yearly')

def load_npy_data_from_vars(data_dir, var_names, key):
    """
    Load temporal data from .npy files for the specified variable names and time key.
    Args:
        data_dir (str): Directory where the .npy files are stored.
        var_names (list): List of variable names to load.
        key (datetime): Time key for which to load the data.
    Returns:
        var_data (np.ndarray): Loaded data array of shape (num_vars, H, W).
    """

    var_data = []

    for var in var_names:
        # print(var, key)
        file_path = os.path.join(data_dir, var, f"{key.strftime('%Y%m%d')}.npy")
        if not os.path.exists(file_path):
            file_path = os.path.join(data_dir, var, f"{key.strftime('%Y%m%d%H')}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = np.load(file_path)
        if "tp" in var:
            # print(var, data.shape, np.min(data), np.max(data), np.mean(data))
            data = np.log(1+data/1e-2)

        # print(var, data.shape)
        data = data[1:] # (120, 240)
        var_data.append(data)
    var_data = np.stack(var_data, axis=0)  # Shape: (num_vars, H, W)
    
    return var_data


def load_npy_multi_data_from_vars(data_dir, var_names, key, idx_of_day=None):
    """
    Load temporal data from .npy files for the specified variable names and time key.
    Args:
        data_dir (str): Directory where the .npy files are stored.
        var_names (list): List of variable names to load.
        key (datetime): Time key for which to load the data.
    Returns:
        var_data (np.ndarray): Loaded data array of shape (num_vars, H, W).
    """
    time_var_data = []
    for idx in range(6, 25, 6):
        var_data = []
        idx_key = key + pd.Timedelta(hours=idx)
        # print(idx_key)
        for var in var_names:
            file_path = os.path.join(data_dir, var, f"{idx_key.strftime('%Y%m%d%H')}.npy")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            try:
                data = np.load(file_path)
            except Exception as e:
                raise FileNotFoundError(f"Reading file not found: {file_path}")
            if "tp" in var:
                # print(var, data.shape, np.min(data), np.max(data), np.mean(data))
                data = np.log(1+data/1e-2)

            # print(var, data.shape)
            data = data[1:] # (120, 240)
            var_data.append(data)
        var_data = np.stack(var_data, axis=0)  # Shape: (num_vars, H, W)
        time_var_data.append(var_data)
    time_var_data = np.stack(time_var_data, axis=0) # Shape: (len_time_idx, num_vars, H, W)
    
    if idx_of_day is not None:
        time_var_data = time_var_data[idx_of_day]
    
    return time_var_data

class Dataset_v1(Dataset):
    def __init__(self, data_dir, times, 
                 var_names=["t2m", "msl", "tp_gpm"],
                 lat=np.arange(90, -90-0.25, -0.25),
                 lon=np.arange(0, 360, 0.25),
        ):
        """
        Args:
            data (list): List of data samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data_dir = data_dir
        self.var_names = var_names
        self.lat = lat
        self.lon = lon

        self.key_times = list(pd.date_range(start=times[0], end=times[1], freq='D'))

        mean, std = get_all_vars_mean_std(var_names)
        self.vars_transform = transforms.Normalize(mean=mean, std=std)

        # mean, std = get_all_vars_mean_std([tp_var])
        # self.tp_transform = transforms.Normalize(mean=mean, std=std)
        
    def __len__(self):
        return len(self.key_times)

    def __getitem__(self, idx):
        
        key = self.key_times[idx]

        input_data = load_npy_data_from_vars(
            self.data_dir, self.var_names, key, 
        )

        target_data = load_npy_data_from_vars(
            self.data_dir, self.var_names, key + pd.Timedelta(days=1)
        )

        # Normalize input data
        input_data = self.vars_transform(torch.tensor(input_data, dtype=torch.float32))
        target_data = self.vars_transform(torch.tensor(target_data, dtype=torch.float32))
        
        return input_data, target_data, key.strftime('%Y%m%d')
    


class Dataset_v2(Dataset):
    def __init__(self, data_dir, times, 
                 var_names=["t2m", "msl", "tp_gpm"],
                 lat=np.arange(90, -90-0.25, -0.25),
                 lon=np.arange(0, 360, 0.25),
        ):
        """
        Args:
            data (list): List of data samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data_dir = data_dir
        self.var_names = var_names
        self.lat = lat
        self.lon = lon

        self.key_times = list(pd.date_range(start=times[0], end=times[1], freq='D'))

        mean, std = get_all_vars_mean_std(var_names)
        self.vars_transform = transforms.Normalize(mean=mean, std=std)

        # mean, std = get_all_vars_mean_std([tp_var])
        # self.tp_transform = transforms.Normalize(mean=mean, std=std)
        
    def __len__(self):
        return len(self.key_times)

    def __getitem__(self, idx):
        
        key = self.key_times[idx]

        input_data_t1 = load_npy_data_from_vars(
            self.data_dir, self.var_names, key - pd.Timedelta(days=1), 
        )

        input_data_t2 = load_npy_data_from_vars(
            self.data_dir, self.var_names, key, 
        )

        target_data = load_npy_data_from_vars(
            self.data_dir, self.var_names, key + pd.Timedelta(days=1)
        )

        # Normalize input data
        input_data_t1 = self.vars_transform(torch.tensor(input_data_t1, dtype=torch.float32))
        input_data_t2 = self.vars_transform(torch.tensor(input_data_t2, dtype=torch.float32))
        target_data = self.vars_transform(torch.tensor(target_data, dtype=torch.float32))
        
        input_data = torch.stack([input_data_t1, input_data_t2], dim=0)  # (2, var, H, W)
        target_data = target_data.unsqueeze(0)  # (1, var, H, W)
        return input_data, target_data, key.strftime('%Y%m%d')


class Dataset_v3(Dataset):
    def __init__(self, data_dir, times, 
                 var_names=["t2m", "msl"],
                 lat=np.arange(90, -90-0.25, -0.25),
                 lon=np.arange(0, 360, 0.25),
                 tp_var="tp_gpm",
        ):
        """
        Args:
            data (list): List of data samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data_dir = data_dir
        self.var_names = var_names
        self.tp_var = [tp_var]
        self.lat = lat
        self.lon = lon

        self.key_times = list(pd.date_range(start=times[0], end=times[1], freq='D'))

        mean, std = get_all_vars_mean_std(var_names)
        tp_mean, tp_std = get_all_vars_mean_std(self.tp_var)
        self.vars_transform = transforms.Normalize(mean=mean, std=std)
        self.tp_transform = transforms.Normalize(mean=tp_mean, std=tp_std)
        
    def __len__(self):
        return len(self.key_times)

    def __getitem__(self, idx):
        
        key = self.key_times[idx]

        input_data = load_npy_data_from_vars(
            self.data_dir, self.var_names, key, 
        )

        tp_data = load_npy_data_from_vars(
            self.data_dir, self.tp_var, key
        )

        # Normalize input data
        input_data = self.vars_transform(torch.tensor(input_data, dtype=torch.float32))
        tp_data = self.tp_transform(torch.tensor(tp_data, dtype=torch.float32))
        
        return input_data, tp_data, key.strftime('%Y%m%d')
    
class Dataset_MultiTime_v3(Dataset):
    def __init__(self, data_dir, times, 
                 var_names=["t2m", "msl"],
                 lat=np.arange(90, -90-0.25, -0.25),
                 lon=np.arange(0, 360, 0.25),
                 tp_var="tp_gpm",
                 idx_of_day=[0, 1, 2, 3],
        ):
        """
        Args:
            data (list): List of data samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data_dir = data_dir
        self.var_names = var_names
        self.tp_var = [tp_var]
        self.lat = lat
        self.lon = lon
        self.idx_of_day = idx_of_day

        self.key_times = list(pd.date_range(start=times[0], end=times[1], freq='D'))

        mean, std = get_all_vars_mean_std(var_names)
        tp_mean, tp_std = get_all_vars_mean_std(self.tp_var)
        self.vars_transform = transforms.Normalize(mean=mean, std=std)
        self.tp_transform = transforms.Normalize(mean=tp_mean, std=tp_std)
        
    def __len__(self):
        return len(self.key_times)

    def __getitem__(self, idx):
        
        key = self.key_times[idx]

        input_data = load_npy_multi_data_from_vars(
            self.data_dir, self.var_names, key, self.idx_of_day
        )

        tp_data = load_npy_data_from_vars(
            self.data_dir, self.tp_var, key
        )

        # Normalize input data
        input_data = self.vars_transform(torch.tensor(input_data, dtype=torch.float32))
        tp_data = self.tp_transform(torch.tensor(tp_data, dtype=torch.float32))
        
        return input_data, tp_data, key.strftime('%Y%m%d')

def get_var_names(surf_vars, atmo_vars, levels):
    var_names = surf_vars.copy()
    for var in atmo_vars:
        for level in levels:
            var_names.append(f"{var}_{level}")
    return var_names

if __name__ == "__main__":

    # data_dir = '/home-ssd/Users/gm_intern/liguowen/data/ERA5/daily_npy/1p5deg/data'
    data_dir = '/home-ssd/Users/gm_intern/liguowen/data/ERA5/1p5deg/npy_data'
    year_list = range(2004, 2024)
    # all_PS, all_ACC, all_RMSE = [], [], []
    tp_var = 'gpm_tp_24hr'
    for year in year_list:
        times = [f'{year}-01-01', f'{year}-12-31']

        era5_surface_vars = ['t2m', 'msl', 'u10', 'v10']
        # era5_upper_vars = ['t', 'z', 'u', 'v', 'q']
        era5_upper_vars = ['t', 'z', 'q']
        # levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        levels = [850, 700, 500]
        var_names = get_var_names(era5_surface_vars, era5_upper_vars, levels)
        # dataset = Dataset_v2(data_dir, times, var_names=var_names)
        dataset = Dataset_MultiTime_v3(data_dir, times, var_names=var_names, tp_var=tp_var)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
        
        mean, std = get_all_vars_mean_std(var_names)
        mean_denorm, std_denorm = -mean / std, 1 / std
        transform_denorm = transforms.Normalize(mean=mean_denorm, std=std_denorm)

        tp_mean , tp_std = get_all_vars_mean_std([tp_var])
        tp_mean_denorm, tp_std_denorm = -tp_mean / tp_std, 1 / tp_std
        transform_tp_denorm = transforms.Normalize(mean=tp_mean_denorm, std=tp_std_denorm)

        start_time = time.time()
        
        for i, (input_data, target_data, time_str) in enumerate(dataloader):
            print(f"Batch {i}:")
            # for j, temp_data in enumerate(temporal_data_list):
            #     if temp_data is not None:
            #         print(f"  Temporal Data {j} shape: {temp_data.shape}")
            # print(f"  Input Data shape: {input_data.shape}")
            # print(f"  Target Data shape: {target_data.shape}")
            # print(f"  Denormalized Input Data stats - mean: {input_data.mean().item()}, max: {input_data.max().item()}, min: {input_data.min().item()}, std: {input_data.std().item()}")
            # print(f" Denormalized Target Data stats - mean: {target_data.mean().item()}, max: {target_data.max().item()}, min: {target_data.min().item()}, std: {target_data.std().item()}")
            # for idx, var in enumerate(var_names):
            #     if input_data.dim() == 5:
            #         print(f"    Var: {var}", input_data[:, :, idx].mean().item(), input_data[:, :, idx].std().item())
            #     elif input_data.dim() == 4:
            #         print(f"    Var: {var}", input_data[:, idx].mean().item(), input_data[:, idx].std().item())
            input_data = transform_denorm(input_data)
            target_data = transform_tp_denorm(target_data)
            target_data = (torch.exp(target_data) - 1) * 1e-2
            # tp_idx = var_names.index(tp_var)
            # if input_data.dim() == 5 and tp_idx >= 0:
            #     input_data[:, :, tp_idx, :, :] = (torch.exp(input_data[:, :, tp_idx, :, :]) - 1) * 1e-2
            # elif input_data.dim() == 4 and tp_idx >= 0:
            #     input_data[:, tp_idx, :, :] = (torch.exp(input_data[:, tp_idx, :, :]) - 1) * 1e-2
            # else:
            #     target_data = (torch.exp(target_data) - 1) * 1e-2
            # print(f"  Denormalized Input data stats (after exp) - mean: {input_data.mean().item()}, max: {input_data.max().item()}, min: {input_data.min().item()}, std: {input_data.std().item()}")
            # print(f" Denormalized Target Tp stats (after exp) - mean: {target_data.mean().item()}, max: {target_data.max().item()}, min: {target_data.min().item()}, std: {target_data.std().item()}")
            # input_data = input_data.cpu().numpy()
            print(f"  Time taken: {time.time() - start_time} seconds")
            i += 1
            # if i == 1:
            #     break