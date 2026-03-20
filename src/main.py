import torch
import random
import numpy as np
import os
import warnings
import time
import torch.distributed as dist

from collections import Counter
from tqdm import tqdm
from model import Model_Loader
from model.model_utils import make_data_parallel, get_num_parameters
from losses_metrics import Losses,TTLosses,Metrics, TTMetrics
from config.config import parse_configs
from utils.logger import Logger
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint, reduce_tensor, to_python_float, print_nvidia_driver_version
from utils.misc import AverageMeter, ProgressMeter, print_gpu_memory_usage
from data_process.dataloader import  create_occlusion_train_val_dataloader, create_occlusion_test_dataloader
from torch.utils.tensorboard import SummaryWriter


# torch.autograd.set_detect_anomaly(True)

def main():
    configs = parse_configs()

    rank = int(os.environ.get("RANK", 0))  # Default to 0 if RANK is not set

    if torch.cuda.is_available() and rank==0:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.cuda.manual_seed_all(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    main_worker(configs)



def main_worker(configs):

    configs.rank = int(os.environ["RANK"])
    configs.world_size = int(os.environ["WORLD_SIZE"])
    # Set the GPU for this process
    configs.gpu_idx = int(os.environ["LOCAL_RANK"])  # Rank within the current node
    configs.device = torch.device(f'cuda:{configs.gpu_idx}')

    print(f"Running on rank {configs.rank}, using GPU {configs.gpu_idx}")

    if configs.gpu_idx is not None:
        print("Use GPU: {} for training".format(configs.gpu_idx))
        configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))


    if configs.distributed:
        dist.init_process_group(
            backend=configs.dist_backend,
            init_method=configs.dist_url,
            world_size=configs.world_size,
            rank=configs.rank
        )
        

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        logger = Logger(configs.logs_dir, 'train')
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None
    
    model = Model_Loader(configs).load_model()
    # Move model to device before broadcasting
    model = model.to(configs.device)
    
    print(f"Rank {configs.rank}: Model built with {sum(p.numel() for p in model.parameters())} parameters.")

    try:
        model = make_data_parallel(model, configs)
        print("Model made parallel successfully.")
    except RuntimeError as e:
        print(f"Runtime error during parallelization: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    scaler = torch.amp.GradScaler()
    best_val_loss = np.inf
    earlystop_count = 0
    is_best = False

    loss_func = Losses(configs=configs, loss_type=configs.loss_function, device=configs.device)
    

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader, option with normal data
    
    train_loader, val_loader, train_sampler = create_occlusion_train_val_dataloader(configs, subset_size=configs.num_samples)

    test_loader = None
    if configs.no_test == False:
        test_loader = create_occlusion_test_dataloader(configs, configs.num_samples)

    if logger is not None:
        batch_data, _ = next(iter(train_loader))
        logger.info(f"Batch data shape: {batch_data.shape}")

    # Print the number of samples for this GPU/worker
    if configs.distributed:
        print(f"GPU {configs.gpu_idx} (Rank {configs.rank}): {len(train_loader.dataset)} samples total, {len(train_loader.sampler)} samples for this GPU")
    
    if logger is not None:
        logger.info('number of batches in train set: {}'.format(len(train_loader)))
        if val_loader is not None:
            logger.info('number of batches in val set: {}'.format(len(val_loader)))
        if test_loader is not None:
            logger.info('number of batches in test set: {}'.format(len(test_loader)))

    
    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        # Get the current learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}] learning rate: {:.2e}'.format(epoch, configs.num_epochs, lr))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
    
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_func, scaler, epoch, configs, logger)
        loss_dict = {'train': train_loss}

        if configs.no_val == False:
            val_loss = evaluate_one_epoch(val_loader, model, loss_func, epoch, configs, logger)
            is_best = val_loss <= best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            loss_dict['val'] = val_loss

        if configs.no_test == False:
            test_loss = evaluate_one_epoch(test_loader, model, loss_func, epoch, configs, logger)
            loss_dict['test'] = test_loss
        # Write tensorboard
        if tb_writer is not None:
            tb_writer.add_scalars('Loss', loss_dict, epoch)
        # Save checkpoint
        if configs.is_master_node:
            saved_state = get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss,
                                        earlystop_count)
            
            # Save checkpoint if it's the best
            if is_best:
                save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best=True, epoch=epoch)
                logger.info(f"Best checkpoint has been saved at epoch {epoch}")
            
            # Save checkpoint based on checkpoint frequency
            if (epoch % configs.checkpoint_freq) == 0:
                save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best=False, epoch=epoch)
                logger.info(f"Checkpoint has been saved at epoch {epoch}")
        # Check early stop training
        if configs.earlystop_patience is not None:
            earlystop_count = 0 if is_best else (earlystop_count + 1)
            print_string = ' |||\t earlystop_count: {}'.format(earlystop_count)
            if configs.earlystop_patience <= earlystop_count:
                print_string += '\n\t--- Early stopping!!!'
                break
            else:
                print_string += '\n\t--- Continue training..., earlystop_count: {}'.format(earlystop_count)
            if logger is not None:
                logger.info(print_string)
        # Adjust learning rate
        if configs.lr_type == 'plateau':
            if configs.no_val:
                lr_scheduler.step(test_loss)
            else:
                lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()

def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_loader, model, optimizer, loss_func, scaler, epoch, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    # switch to train mode
    model.train()
    start_time = time.time()
    
    for batch_idx, (batch_data, (_, labels, visibiltity, status)) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - start_time)

        batch_size = batch_data.size(0)
        batch_data = batch_data.to(configs.device, dtype=torch.float)
        labels = labels.to(configs.device, dtype=torch.float)
        visibiltity = visibiltity.to(configs.device)

        if configs.model_choice in ['tracknet', 'tracknetv2', 'wasb', 'monoTrack', 'TTNet', 'tracknetv4']:
            # #for tracknet we need to rehsape the data
            B, N, C, H, W = batch_data.shape
            # Permute to bring frames and channels together
            batch_data = batch_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
            # Reshape to combine frames into the channel dimension
            batch_data = batch_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]

        with torch.autocast(device_type='cuda'):
            output_heatmap = model(batch_data) # output in shape [B,H,W],if TTNet, output is ([B,W], [B,H])
            output_heatmap = output_heatmap.float()

        total_loss = loss_func(output_heatmap, labels, visibiltity)

        # For torch.nn.DataParallel case
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)

        # zero the parameter gradients
        optimizer.zero_grad()
        # compute gradient and perform backpropagation
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if configs.distributed:
            reduced_loss = reduce_tensor(total_loss, configs.world_size)
        else:
            reduced_loss = total_loss
        losses.update(to_python_float(reduced_loss), batch_size)
        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)
        # Log message
        if logger is not None:
            if ((batch_idx + 1) % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))
        start_time = time.time()

    return losses.avg

def evaluate_one_epoch(val_loader, model, loss_func, epoch, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    rmses = AverageMeter('RMSE', ':.4e')
    accuracy_overall = AverageMeter('Accuracy', ':6.4f')
    precision_overall = AverageMeter('Precision', ':6.4f')
    recall_overall = AverageMeter('Recall', ':6.4f')
    f1_overall = AverageMeter('F1', ':6.4f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time, losses, rmses, accuracy_overall, precision_overall, recall_overall, f1_overall],
                             prefix="Evaluate - Epoch: [{}/{}]".format(epoch, configs.num_epochs))
    # switch to evaluate mode
    model.eval()
    metrics = Metrics(configs=configs, device=configs.device)

    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (batch_data, (_, labels, visibility, status)) in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            batch_size = batch_data.size(0)

            batch_data = batch_data.to(configs.device, dtype=torch.float)
            labels = labels.to(configs.device, dtype=torch.float)
            visibility = visibility.to(configs.device)

            if configs.model_choice in ['tracknet', 'tracknetv2', 'wasb', 'monoTrack', 'TTNet', 'tracknetv4']:
                # #for tracknet we need to rehsape the data
                B, N, C, H, W = batch_data.shape
                # Permute to bring frames and channels together
                batch_data = batch_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
                # Reshape to combine frames into the channel dimension
                batch_data = batch_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]

            with torch.autocast(device_type='cuda'):
                output_heatmap = model(batch_data) # output in shape ([B, H, W]) if output heatmap, just raw logits
                output_heatmap = output_heatmap.float()

            total_loss = loss_func(output_heatmap, labels, visibility)

            mse, rmse, mae, euclidean_distance = metrics.calculate_metrics(output_heatmap, labels)
            post_processed_coords = metrics.extract_coordinates(output_heatmap)
            precision, recall, f1, accuracy = metrics.precision_recall_f1(post_processed_coords, labels)

            rmse_tensor = torch.tensor(rmse).to(configs.device)
            precision_tensor = torch.tensor(precision).to(configs.device)
            recall_tensor = torch.tensor(recall).to(configs.device)
            f1_tensor = torch.tensor(f1).to(configs.device)
            accuracy_tensor = torch.tensor(accuracy).to(configs.device)
           
            # For torch.nn.DataParallel case
            if (not configs.distributed) and (configs.gpu_idx is None):
                total_loss = torch.mean(total_loss)

            if configs.distributed:
                reduced_loss = reduce_tensor(total_loss, configs.world_size)
                reduced_rmse = reduce_tensor(rmse_tensor, configs.world_size)
                reduced_accuracy = reduce_tensor(accuracy_tensor, configs.world_size)
                reduced_precision = reduce_tensor(precision_tensor, configs.world_size)
                reduced_recall = reduce_tensor(recall_tensor, configs.world_size)
                reduced_f1 = reduce_tensor(f1_tensor, configs.world_size)
                
            else:
                reduced_rmse = rmse
                reduced_accuracy = accuracy
                reduced_precision = precision
                reduced_recall = recall
                reduced_f1 = f1
                reduced_loss = total_loss

            losses.update(to_python_float(reduced_loss), batch_size)
            rmses.update(to_python_float(reduced_rmse), batch_size)
            accuracy_overall.update(to_python_float(reduced_accuracy), batch_size)
            precision_overall.update(to_python_float(reduced_precision), batch_size)
            recall_overall.update(to_python_float(reduced_recall), batch_size)
            f1_overall.update(to_python_float(reduced_f1), batch_size)

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()
    if logger is not None:
        logger.info(f"overall evaluation performance loss:{losses.avg}, accuracy: {accuracy_overall.avg}")
    return losses.avg

if __name__ == '__main__':
    main()
    print("complete building detector")


