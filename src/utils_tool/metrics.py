# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import lpips  # https://github.com/richzhang/PerceptualSimilarity
from sklearn.metrics import f1_score, confusion_matrix

def mse(pred, y, vars, lat=None, mask=None):
    """Mean squared error

    Args:
        pred: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """

    loss = (pred - y) ** 2

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (loss[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = loss[:, i].mean()

    if mask is not None:
        loss_dict["loss"] = (loss.mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = loss.mean(dim=1).mean()

    return loss_dict

def lat_weighted_mse(pred, y, vars, lat, mask=None, weight=None):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    if weight is not None:
        error = (pred - y) ** 2 * weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(pred.device)
    else:
        error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    # print("w_lat", w_lat.shape)
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()
    # print("error", error.shape)
    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        if weight is not None:
            loss_dict["loss"] = (error * w_lat.unsqueeze(1)).sum(dim=1).mean()
        else:
            loss = 0.
            loss1 = (error[:, 0] * w_lat).mean()
            for i, var in enumerate(vars):
                loss += (error[:, i] * w_lat).mean() / ((error[:, i] * w_lat).mean() / loss1).detach()

            # loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()
            loss_dict["loss"] = loss

    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    pred = transform(pred)
    y = transform(y)

    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    clim = clim.to(device=y.device).unsqueeze(0)

    pred = pred - clim
    y = y - clim 
    
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def lat_weighted_rmse_tp(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def rmse_tp(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i], dim=(-2, -1)))
            )

    loss_dict["rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_acc_tp(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict



#  D.3.1 Learned Perceptual Image Patch Similarity
_lpips_alex = lpips.LPIPS(net='alex').eval()  # create one global LPIPS evaluator

def lpips_metric(pred, y, transform=None, vars=None, lat=None, clim=None, log_postfix=""):
    """Learned Perceptual Image Patch Similarity (LPIPS)."""
    if transform is not None:
        pred = transform(pred)
        y    = transform(y)
    # flatten B×V into batch of images
    B, V, H, W = pred.shape
    pred_imgs = pred.reshape(B*V, 1, H, W).repeat(1,3,1,1)  # LPIPS expects 3 channels
    y_imgs    = y.reshape(B*V, 1, H, W).repeat(1,3,1,1)
    with torch.no_grad():
        lpips_vals = _lpips_alex(pred_imgs, y_imgs)
    # reshape back and average per variable
    lpips_vals = lpips_vals.view(B, V).mean(dim=0)
    return {f"LPIPS_{vars[i]}_{log_postfix}": lpips_vals[i].item() for i in range(V)}

_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # adjust data_range if you normalize differently

def ssim_metric(pred, y, transform=None, vars=None, lat=None, clim=None, log_postfix=""):
    """Structural Similarity Index Measure (SSIM)."""
    if transform is not None:
        pred = transform(pred)
        y    = transform(y)
    # compute SSIM per channel
    B, V, H, W = pred.shape
    res = {}
    for i, v in enumerate(vars):
        # torchmetrics expects shape [B,1,H,W]
        si = _ssim(pred[:,i:i+1], y[:,i:i+1]).item()
        res[f"SSIM_{v}_{log_postfix}"] = si
    return res


def spatial_rmse(pred, y, transform=None, vars=None, lat=None, clim=None, log_postfix=""):
    """Return per‐variable, per‐grid‐cell RMSE map (as a tensor)."""
    if transform: 
        pred = transform(pred)
        y    = transform(y)
    err2 = (pred - y).pow(2).mean(dim=0)  # [V,H,W]
    rmse = torch.sqrt(err2)
    return {f"spatial_rmse_{vars[i]}_{log_postfix}": rmse[i] for i in range(len(vars))}


def f1_metric(pred, y, transform=None, vars=None, lat=None, clim=None, log_postfix="", threshold=0.1):
    """
    Global F1 score after thresholding. 
    For each grid cell we treat pred>=threshold as "event".
    """
    if transform:
        pred = transform(pred)
        y    = transform(y)
    B, V, H, W = pred.shape
    res = {}
    pred_bin = (pred >= threshold).cpu().numpy().reshape(-1)
    y_bin    = (y    >= threshold).cpu().numpy().reshape(-1)
    f1 = f1_score(y_bin, pred_bin, zero_division=0)
    res[f"F1_{log_postfix}"] = f1
    return res

def ts_metric(pred, y, transform=None, vars=None, lat=None, clim=None, log_postfix="", threshold=0.1):
    """
    TS = hits / (hits + misses + false alarms)
    """
    if transform:
        pred = transform(pred)
        y    = transform(y)
    P = pred >= threshold
    O = y    >= threshold
    hits   = (P & O).sum().item()
    misses = (~P & O).sum().item()
    falses = (P & ~O).sum().item()
    ts = hits / (hits + misses + falses) if (hits+misses+falses)>0 else 0.0
    return {f"TS_{log_postfix}": ts}

def hss_metric(pred, y, transform=None, vars=None, lat=None, clim=None, log_postfix="", threshold=0.1):
    """
    HSS = (2*(AD - BC)) / ((A+C)*(C+D)+(A+B)*(B+D))
    where A=hits, B=false alarms, C=misses, D=correct negatives
    """
    if transform:
        pred = transform(pred)
        y    = transform(y)
    P = pred >= threshold
    O = y    >= threshold
    A = (P & O).sum().item()
    B = (P & ~O).sum().item()
    C = (~P & O).sum().item()
    D = (~P & ~O).sum().item()
    num = 2*(A*D - B*C)
    den = (A+C)*(C+D)+(A+B)*(B+D)
    hss = num/den if den>0 else 0.0
    return {f"HSS_{log_postfix}": hss}

# D.3.9 Energy Conservation Metric (ECM)
def ecm_metric(pred, y, transform=None, vars=None, lat=None, clim=None, log_postfix=""):
    """
    ECM = 1 - | sum(pred) - sum(obs) | / sum(obs)
    measures bulk conservation across all grid cells & variables.
    """
    if transform:
        pred = transform(pred)
        y    = transform(y)
    total_p = pred.sum().item()
    total_o = y.sum().item()
    ecm = 1 - abs(total_p - total_o)/total_o if total_o!=0 else 0.0
    return {f"ECM_{log_postfix}": ecm}
