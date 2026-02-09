from typing import Iterable
import torch
import utils
from tqdm import tqdm
import sys
import os
try:
    from ..data.mean_std import get_all_vars_mean_std
except ImportError:
    from data.mean_std import get_all_vars_mean_std
from torchvision import transforms
import numpy as np
import time


def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device,  log_writer=None, epoch: int = 0, epochs: int = 1,
                    task_name=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    train_pbar = tqdm(data_loader, desc=f"{task_name} Epoch {epoch+1}/{epochs}")
    for data_iter_step, batch in enumerate(train_pbar):

        inp_data, out_data, _ = batch
        del batch

        optimizer.zero_grad()
        loss, ctx = model(inp_data, out_data)
        
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        # Calculate and record the minimum and maximum learning rates
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        
        # Record the weight decay value
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")

            log_writer.set_step()
        # break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
