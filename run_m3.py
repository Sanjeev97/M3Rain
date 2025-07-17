import argparse
import json
import math
import sys
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory

from models.m3 import M3
import util.misc as misc
from dataset.data_wrapper import DataWrapper
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from typing import Iterable
import util.lr_sched as lr_sched


from util import metrics

# Import the new H5 data loader
from dataset.aligned_dataset import AlignedH5Dataset, create_data_loaders

from datetime import datetime

torch.manual_seed(0)
np.random.seed(0)

# RMSE, R_Squared, Corr
best_metrics = [float("inf"), 0, 0]


def get_args_parser():
    parser = argparse.ArgumentParser('M3 training', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--embed_dim', default=512, type=int, help='embed dimensions')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--model_pvt', default='pvt_tiny', type=str, metavar='MODEL',
                        help='Name of backbone model to train')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # H5 dataset path
    parser.add_argument('--h5_file', type=str, default='/home/C00535626/Radar-Rainfall/M3/data/klch_radar_pws_aligned_100km_20P_composite4.h5',
                        help='Path to the HDF5 file containing aligned radar and PWS data')
   
    parser.add_argument('-sf', '--save_freq', type=int, default=20)

    # evaluate
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')


    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Set up logging
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Create data loaders using the new H5 data loader
    data_loader_train, data_loader_val = create_data_loaders(
        args.h5_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem
    )
    
    model = M3(
        image_size=[128, 128],    # 128x128 images
        patch_size=[16, 16],      # 16x16 patches -> 64 patches per image
        ctx_channels=1,           # Grayscale images
        ts_channels=20,           # 20 time series features
        ts_length=4,              # 4 timesteps
        dim=128,                  # Model dimension
        depth=2,                  # 2 layers (reduced for testing)
        heads=4,                  # 4 attention heads
        mlp_ratio=4,              # MLP expansion ratio
        dropout=0.1,              # Dropout
        num_mlp_heads=1,          # Single output head
        decoder_dim=128,          # Decoder dimension
        decoder_depth=2,          # Decoder depth
    )

    model.to(device)

    # Batch size calculation
    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Set up optimizer
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        evaluate(model, data_loader_val, device)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            # evaluate
            evaluate(model, data_loader_val, device)

            misc.save_model(
                args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    data_wrapper = DataWrapper(train=False)

    true_labels = torch.empty(0)
    pred_labels = torch.empty(0)

    total_step = len(data_loader) - 1
    for data_iter_step, (radar_data, pws_data, _) in enumerate(data_loader):
        max_mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print("[ {} / {}]  Max Mem: {}"
              .format(data_iter_step, total_step, f"{max_mem:.0f}"))

        # Process radar data
        x = radar_data[:, :4, :, :, :].to(device, non_blocking=True)
        
        # Process PWS data - using the last 4 values for ground truth as in original code
        z = pws_data[:, :4, :, :].to(device, non_blocking=True)  # PWS data 
        g = pws_data[:, -4:, :, -1].to(device, non_blocking=True)  # Ground truth # last parameter is rainfall rate

        import torch.nn.functional as F
        # Data reshaping and augmentation
        b, t, c, h, w = x.shape
        # Direct resize (most common approach)
        x = F.interpolate(
            x.view(-1, 1, 101, 101),    # Flatten to [256, 1, 101, 101]
            size=(128, 128),             # Target size
            mode='bilinear',             # Interpolation method
            align_corners=False          # PyTorch default
        ).view(b, 4, 1, 128, 128)      # Reshape back

        z = z.squeeze()  # Remove the last dimension if it's 1
        # print(x.shape)
        # print(z.shape)
        
        # Forward pass
        z_hat = model(ctx=x, ts=z)[0]  # Assuming model returns a tuple, take the first element

        # Collect predictions and true labels
        true_labels = torch.cat([true_labels, g.detach().cpu()], dim=0)
        pred_labels = torch.cat([pred_labels, z_hat.detach().cpu()], dim=0)

    # Extract the last column for evaluation
    true_labels = torch.flatten(true_labels[:, -1:], start_dim=0).detach().cpu().numpy() 
    pred_labels = torch.flatten(pred_labels[:, -1:], start_dim=0).detach().cpu().numpy()

    # Calculate metrics
    rmse, r2, corr = metrics.evaluate(true_labels, pred_labels)

    # Update best metrics
    global best_metrics
    best_metrics = [min(best_metrics[0], rmse), max(best_metrics[1], r2), max(best_metrics[2], corr)]
    print("Metrics: RMSE: {}  R_Squared: {}  Corr: {}".format(
        f"{best_metrics[0]:.2f}", f"{best_metrics[1]:.2f}", f"{best_metrics[2]:.2f}"))


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter
    data_wrapper = DataWrapper()

    criterion = torch.nn.MSELoss().to(device)
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    total_step = len(data_loader) - 1
    for data_iter_step, (radar_data, pws_data, _) in enumerate(data_loader):

        max_mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print("Epoch: [{}]  [ {} / {}]  Max Mem: {}"
              .format(epoch, data_iter_step, total_step, f"{max_mem:.0f}"))

        # Per iteration lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Process radar data
        x = radar_data[:, :4, :, :, :].to(device, non_blocking=True)
        
        # Process PWS data - using the last 4 values for ground truth as in original code
        z = pws_data[:, :4, :, :].to(device, non_blocking=True)  # PWS data 
        g = pws_data[:, -4:, :, -1].to(device, non_blocking=True)  # Ground truth - last 4 values
        # print("ground truth shape:", g.shape)
        g= g.squeeze()  # Remove the last dimension if it's 1
        # print("ground truth shape after squeeze:", g.shape)

        # Data reshaping and augmentation
        b, t, c, h, w = x.shape
        
        #Transform radar data to 128*128
        # x = rearrange(x, 'b t c h w -> (b t) c h w')
        # print(x.shape)
        # print(type(x))
        
        import torch.nn.functional as F
        
        # Direct resize (most common approach)
        x = F.interpolate(
            x.view(-1, 1, 101, 101),    # Flatten to [256, 1, 101, 101]
            size=(128, 128),             # Target size
            mode='bilinear',             # Interpolation method
            align_corners=False          # PyTorch default
        ).view(b, 4, 1, 128, 128)      # Reshape back

        z = z.squeeze()  # Remove the last dimension if it's 1
        # print(x.shape)
        # print(z.shape)
        
        # Forward pass
        z_hat = model(ctx=x, ts=z)[0]  # Assuming model returns a tuple, take the first element
        
        # print("z_hat shape:", z_hat.shape)

        # Compute loss
        # print("Ground truth shape:", g.shape)
        # print("Predicted shape:", z_hat.shape)
        loss = criterion(g, z_hat.squeeze())  # Squeeze to match dimensions
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Gradient accumulation
        # loss /= accum_iter
        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Log metrics
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # TensorBoard logging
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
