import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import time
from pathlib import Path
import datetime
import torch
import torch.backends.cudnn as cudnn
import json
import argparse
import utils
from data.dataset import Dataset_MultiTime

from optim_factory import create_optimizer
from arch import ResCast
from engine import train_one_epoch


def get_args():
    parser = argparse.ArgumentParser(description='Train MoE Model')
    parser.add_argument('-t', '--task_name', type=str, default='test', 
                        help='Name of the task')
    parser.add_argument('--data_dir', type=str, default='path/to/your/data',
                        help="Directory containing .npy files")
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--val_batch_size', default=24, type=int)
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs for training')
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    
    parser.add_argument('--embed_dim', type=int, default=256, 
                        help='Embedding dimension for the model')
    parser.add_argument('-nw', '--num_workers', type=int, default=1, 
                        help='Number of workers for data loading')
    parser.add_argument('--test_years', type=int, nargs='+', default=[2014, 2020, 2022], 
                        help='list of years for testing')
    parser.add_argument('--val_years', type=int, nargs='+', default=[2011, 2015, 2018, 2019, 2023], 
                        help='list of years for validation')
    parser.add_argument('--tp_var', type=str, default='tp_gpm', 
                        help='Variable name for total precipitation')
    parser.add_argument('--idx_of_day', type=int, nargs='+', default=[0, 1, 2, 3], 
                        help='idx list of time in day')
    
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to load checkpoints')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate for the optimizer')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
                        weight decay. We use a cosine schedule for WD and using a larger decay by
                        the end of training improves performance for ViTs.""")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    return parser.parse_args(), None

def save_hyperparameters(config, output_dir, title):
    def make_json_serializable(obj):
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)
        
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
    config_path = os.path.join(output_dir, f"hyperparams_{title}_{timestamp}.json")

    serializable_config = {k: make_json_serializable(v) for k, v in config.items()}

    with open(config_path, "w") as f:
        json.dump(serializable_config, f, indent=4, ensure_ascii=False)

    print(f"Hyperparameters saved to: {config_path}")

def get_var_names(surf_vars, atmo_vars, levels):
    var_names = surf_vars.copy()
    for var in atmo_vars:
        for level in levels:
            var_names.append(f"{var}_{level}")
    return var_names

def main(args, ds_init):

    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Dataset
    era5_surface_vars = ['t2m', 'msl', 'u10', 'v10']
    era5_upper_vars = ['t', 'z', 'u', 'v', 'q']
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    var_names = get_var_names(era5_surface_vars, era5_upper_vars, levels)

    # ==================================================================================
    from torch.utils.data import ConcatDataset

    def build_split_years(val_years, test_years, start=1993, end=2024):
        all_years = list(range(start, end+1))
        val_years = set(val_years)
        test_years = set(test_years)

        train_years = [y for y in all_years if y not in val_years and y not in test_years]

        return train_years, list(val_years), list(test_years)
    
    val_years = args.val_years
    test_years = args.test_years
    train_years, val_years, test_years = build_split_years(val_years, test_years, start=1998, end=2023)

    def year_to_times(year):
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        return [start, end]
    
    train_datasets = [
        Dataset_MultiTime(
            args.data_dir, year_to_times(y),
            var_names=var_names,
            tp_var=args.tp_var,
            idx_of_day=args.idx_of_day
        )
        for y in train_years
    ]

    dataset_train = ConcatDataset(train_datasets)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    print("Sampler_train = %s" % str(sampler_train))


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )


    # =======================================================================================
    select_model = ResCast
    model_config = {
        "var_names": var_names,
        "embed_dim": args.embed_dim,
        "T_in": len(args.idx_of_day)
    } # yaml file

    model = select_model(**model_config)
    model.to(device)
    
    model_without_ddp = model

    # =======================================================================================

    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    optimizer = create_optimizer(args, model_without_ddp)
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))  

    # =======================================================================================

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    start_epoch = 0
    if args.ckpt_path is not None:

        state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(state_dict['model'], strict=True)
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
        tmp_epochs = args.epochs
        tmp_eval = False
        if args.eval: 
            tmp_eval = args.eval
        args = state_dict['args']
        args.eval = tmp_eval
        args.epochs = tmp_epochs

    # =======================================================================================
    config = {}
    config['model'] = select_model
    config = config | vars(args)
    config["var_names"] = var_names
    config['train_years'] = train_years
    config['val_years'] = val_years
    config['test_years'] = test_years
    config["optimizer"] = optimizer

    if args.rank == 0:
        save_hyperparameters(model_config, args.output_dir, "model")
        save_hyperparameters(config, args.output_dir, "train")
    # =======================================================================================
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_time_only = 0
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_start_time = time.time()

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, log_writer=log_writer, 
            epoch=epoch, epochs=args.epochs, task_name=args.task_name
        )
        train_time_only += time.time() - train_start_time
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, 
                    epoch=epoch)
                
        
        log_stats = {'epoch': epoch,
                    **{f'train_{k}': v for k, v in train_stats.items()}}
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    train_time_str = str(datetime.timedelta(seconds=int(train_time_only)))
    print('Training time only {}'.format(train_time_str))
    print('Total time {}'.format(total_time_str))
    
    if args.output_dir and utils.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps('Training time only {}'.format(train_time_str) + "\n"))
            f.write(json.dumps('Total time {}'.format(total_time_str)) + "\n")

    
if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)