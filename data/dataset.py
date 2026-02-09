from torch.utils.data import Dataset
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

class Dataset_MultiTime(Dataset):
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
