# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import torch


NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}
NAME_TO_VAR_14 = {
    "2m_temperature": "t2m_avg",
    "wind_speed_10m": "wind_speed_10m_avg",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr_avg",
    "surface_net_thermal_radiation": "str_avg",
    "boundary_layer_height": "blh_avg",
    "sshf": "sshf_avg",
    "slhf": "slhf_avg",
    "msl": "msl_avg",
    "total_precipitation": "tp_sum",
    "land_sea_mask": "lsm_avg",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z_avg",
    "wind_speed": "wind_speed_avg",
    "temperature": "t_avg",
    "relative_humidity": "r_avg",
    "specific_humidity": "q_avg",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

BOUNDARIES = {
    'Global': {
        'lat_range': (-90, 90),
        'lon_range': (0, 360)
    },
    'China': {
        'lat_range': (0, 55),
        'lon_range': (70, 135)
    },
    'MiddleEast': {
        'lat_range': (10, 40),
        'lon_range': (30, 60)
    },
    'NorthAmerica': {
        'lat_range': (15, 65),
        'lon_range': (220, 300)
    },
    'SouthAmerica': {
        'lat_range': (-55, 20),
        'lon_range': (270, 330)
    },
    'India': {
        'lat_range': (8, 37),
        'lon_range': (68, 97)
    }
}

def get_region_info(region, lat, lon, patch_size):
    region_name = region
    region = BOUNDARIES[region]
    lat_range = region['lat_range']
    lon_range = region['lon_range']
    lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    return {
        'patch_ids': valid_patch_ids,
        'min_h': min_h,
        'max_h': max_h,
        'min_w': min_w,
        'max_w': max_w,
        'region_name': region_name 
    }


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

