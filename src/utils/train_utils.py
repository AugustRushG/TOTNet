import copy
import os
import math

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
import torch.distributed as dist
import subprocess
import time

def print_nvidia_driver_version():
    try:
        # Run nvidia-smi command and capture the output
        nvidia_smi_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        # Extract the driver version from the output
        for line in nvidia_smi_output.split('\n'):
            if 'Driver Version' in line:
                print("NVIDIA Driver Version:")
                print(line.strip())
                return
        print("NVIDIA driver version not found in nvidia-smi output.")
    except FileNotFoundError:
        print("nvidia-smi is not installed or NVIDIA drivers are not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_optimizer(configs, model):
    """Create optimizer for training process"""
    if hasattr(model, 'module'):
        train_params = [param for param in model.module.parameters() if param.requires_grad]
    else:
        train_params = [param for param in model.parameters() if param.requires_grad]

    if configs.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(train_params, lr=configs.lr, momentum=configs.momentum,
                                    weight_decay=configs.weight_decay)
    elif configs.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(train_params, lr=configs.lr, weight_decay=configs.weight_decay)
    elif configs.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(train_params, lr=configs.lr, weight_decay=configs.weight_decay)
    else:
        assert False, "Unknown optimizer type"

    return optimizer


def create_lr_scheduler(optimizer, configs):
    """Create learning rate scheduler for training process"""
    if configs.lr_type == 'step_lr':
        lr_scheduler = StepLR(optimizer, step_size=configs.lr_step_size, gamma=configs.lr_factor)
    elif configs.lr_type == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=configs.lr_factor, patience=configs.lr_patience)
    elif configs.optimizer_type == 'cosin':
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / configs.num_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
        lr_scheduler.last_epoch = configs.start_epoch - 1  # do not move
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        # plot_lr_scheduler(optimizer, scheduler, epochs)
    else:
        raise TypeError

    return lr_scheduler

def get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss, earlystop_count):
    """Get the information to save with checkpoints"""
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    saved_state = {
        'epoch': epoch,
        'configs': configs,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scheduler': lr_scheduler.state_dict(),
        'state_dict': model_state_dict,
        'best_val_loss': best_val_loss,
        'earlystop_count': earlystop_count,
    }

    return saved_state

def save_checkpoint(checkpoints_dir, saved_fn, saved_state, is_best, epoch):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""
    if is_best:
        save_path = os.path.join(checkpoints_dir, '{}_best.pth'.format(saved_fn))
    else:
        save_path = os.path.join(checkpoints_dir, '{}_epoch_{}.pth'.format(saved_fn, epoch))

    torch.save(saved_state, save_path)
    print('save a checkpoint at {}'.format(save_path))

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
    

def benchmark_fps(model, batch_data, device="cuda", num_warmup=10, num_iters=30):
    """
    Benchmark the forward pass FPS of a PyTorch model.

    Args:
        model (torch.nn.Module): The model to benchmark.
        batch_data (torch.Tensor): Input batch. Shape could be
            - [B, C, H, W] for images
            - [B, T, C, H, W] for video clips
        device (str): "cuda" or "cpu".
        num_warmup (int): Number of warmup iterations.
        num_iters (int): Number of timed iterations.

    Returns:
        dict: { "avg_time": float, "fps_frames": float, "fps_clips": float }
    """
    model.eval()
    batch_data = batch_data.to(device)

    # Warmup runs (stabilize GPU/CPU performance)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(batch_data)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timed runs
    total_time = 0.0
    for _ in range(num_iters):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            _ = model(batch_data)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_time = total_time / num_iters  # seconds per forward pass

    # Throughput calculations
    if batch_data.ndim == 4:   # [B, C, H, W] = images
        num_clips = batch_data.shape[0]
        num_frames = num_clips
    elif batch_data.ndim == 5: # [B, T, C, H, W] = video clips
        num_clips = batch_data.shape[0]
        num_frames = num_clips * batch_data.shape[1]
    else:
        raise ValueError("Unsupported input shape for batch_data.")

    fps_frames = num_frames / avg_time
    fps_clips = num_clips / avg_time

    return {
        "avg_time": avg_time,
        "fps_frames": fps_frames,
        "fps_clips": fps_clips
    }