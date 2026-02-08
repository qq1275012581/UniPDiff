import numpy as np
from scipy.interpolate import interp2d
from inspect import isfunction

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def extract_into_tensor(a, t, x_shape, batch_axis=0):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    out_shape = [1, ] * len(x_shape)
    out_shape[batch_axis] = batch_size
    return out.reshape(out_shape)


def resize_array_by_interp2d(input_array, scale_factor):
    """
    Resizes an array by a given scale factor using 2D linear interpolation.

    Parameters:
    - input_array: The input numpy array to resize.
    - scale_factor: The scale factor by which to resize the array.

    Returns:
    - The resized array as a numpy array.
    """
    rows, cols = input_array.shape

    new_rows, new_cols = int(rows * scale_factor), int(cols * scale_factor)

    old_x, old_y = np.linspace(0, 1, cols), np.linspace(0, 1, rows)
    new_x, new_y = np.linspace(0, 1, new_cols), np.linspace(0, 1, new_rows)

    interp_func = interp2d(old_x, old_y, input_array, kind='linear')

    resized_array = interp_func(new_x, new_y)

    return resized_array


def var_to_unit(var_name):
    units = {
        "temperature": "K",
        "wind_speed": "m/s",
        "geopotential": "m",
        "humidity": "g.kg-1"
    }
    for key, value in units.items():
        if key in var_name:
            return value
    raise ValueError(f"Unknown variable name: {var_name}")