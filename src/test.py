import time
import sys
import os
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from utils.logger import Logger
from tqdm import tqdm

sys.path.append('./')

from data_process.dataloader import create_occlusion_test_dataloader, create_occlusion_train_val_dataloader
from model.model_utils import make_data_parallel, get_num_parameters, load_pretrained_model
from model import Model_Loader
from losses_metrics import Metrics
from utils.misc import AverageMeter
from utils.visualization import visualize_and_save_2d_heatmap, save_batch_optical_flow_visualization
from config.config import parse_configs


def main():
    configs = parse_configs()
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx

    if configs.gpu_idx is not None:
        print("Use GPU: {} for training".format(configs.gpu_idx))
        configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    model = Model_Loader(configs).load_model()
    model = make_data_parallel(model, configs)

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        print('number of trained parameters of the model: {}'.format(num_parameters))

        logger = Logger(configs.logs_dir, 'test')
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))

    if configs.pretrained_path is not None:
        model = load_pretrained_model(model, configs.pretrained_path, gpu_idx)
 
    test_loader = create_occlusion_test_dataloader(configs, configs.num_samples)
    print(f"number of batches in test is {len(test_loader)}")
    test(test_loader, model, configs, logger)


def test(test_loader, model, configs, logger):
    # Initialize metrics for each visibility category
    visibility_metrics = { 
        vis: {
            "distance": AverageMeter(f"Distance_Vis_{vis}", ":6.4f"),
            "accuracy": AverageMeter(f"Accuracy_Vis_{vis}", "6.4f"),
        } for vis in range(4)  # Assuming visibility levels 0 to 3
    }

    # Overall metrics
    rmse_overall = AverageMeter('RMSE_Overall', ':6.4f')
    accuracy_overall = AverageMeter('Accuracy', '6.4f')
    precision_overall = AverageMeter('Precision', '6.4f')
    recall_overall = AverageMeter('Recall', '6.4f')
    f1_overall = AverageMeter('F1', '6.4f')
    predictions = []
    labels_list = []

    # calculate fps rate
    total_time = 0
    total_frames = 0
    metrics = Metrics(configs, configs.device)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (batch_data, (_, labels, visibility, _)) in enumerate(tqdm(test_loader)):
            print(f'\n===================== batch_idx: {batch_idx} ================================')
            batch_size = batch_data.size(0)
            batch_data = batch_data.to(configs.device)
            num_frames = batch_data.size(1)
            labels = labels.to(configs.device)
            visibility = visibility.to(configs.device)


            # Compute output
            if configs.model_choice in ['tracknet', 'tracknetv2', 'wasb', 'monoTrack', 'TTNet', 'tracknetv4']:
                # #for tracknet we need to rehsape the data
                B, N, C, H, W = batch_data.shape
                # Permute to bring frames and channels together
                batch_data = batch_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
                # Reshape to combine frames into the channel dimension
                batch_data = batch_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]

            start_time = time.perf_counter()  # Start timing
            output_heatmap = model(batch_data.float())
            end_time = time.perf_counter()  # End timing
            
            total_time += end_time - start_time  # Accumulate time

            mse, rmse, _, _ = metrics.calculate_metrics(output_heatmap, labels)
            post_processed_coords = metrics.extract_coordinates(output_heatmap)
            precision, recall, f1, accuracy = metrics.precision_recall_f1(
                post_processed_coords, labels
            )
            # Store predictions and labels for overall metrics
            predictions.append(post_processed_coords.cpu())
            labels_list.append(labels.cpu())

            # Update metrics for each sample by visibility
            for sample_idx in range(batch_size):
               
                vis_label = visibility[sample_idx].item()  # Visibility label for this sample
                label = labels[sample_idx]
                pred_coords = post_processed_coords[sample_idx]
                x_pred, y_pred = pred_coords[0], pred_coords[1]

                sample_rmse = metrics.calculate_rmse(label[0], label[1], x_pred, y_pred)

                # Check if prediction is within threshold for accuracy
                dist = torch.sqrt((pred_coords[0] - label[0])**2 + (pred_coords[1] - label[1])**2)
          
                ball_size = configs.ball_size*2 if vis_label==3 else configs.ball_size
                within_threshold = dist <= ball_size

                sample_accuracy = 1 if within_threshold else 0
        

                # Update visibility-specific metrics
                visibility_metrics[vis_label]["distance"].update(sample_rmse)
                visibility_metrics[vis_label]["accuracy"].update(sample_accuracy)
                logger.info(f"""Visibility {vis_label} - Ball Detection - Overall: (x, y) - org: ({label[0].item()}, {label[1].item()}), prediction = ({x_pred.item()}, {y_pred.item()}, distance is {sample_rmse:.4f})""")


            # Update overall metrics
            rmse_overall.update(rmse)
            accuracy_overall.update(accuracy)
            precision_overall.update(precision)
            recall_overall.update(recall)
            f1_overall.update(f1)

    predictions = torch.cat(predictions, dim=0)
    labels_list = torch.cat(labels_list, dim=0)

    pck_results = metrics.calculate_pck(
        predictions, labels_list, thresholds=[1, 2, 3, 4, 5]
    )

    total_frames = len(test_loader)*batch_size

    fps = total_frames / total_time if total_time > 0 else 0

    # Print results for each visibility category
    logger.info("===== Visibility-Specific Results =====")
    for vis_label, metrics in visibility_metrics.items():
        logger.info(
            f"Visibility {vis_label}: Distance: {metrics['distance'].avg:.4f}, "
            f"Accuracy: {metrics['accuracy'].avg:.4f}"
        )

    
    # Print overall results
    logger.info("===== Overall Results =====")
    logger.info(
        f"Overall Results: RMSE: {rmse_overall.avg:.4f}, Accuracy: {accuracy_overall.avg:.4f}, \n"
        f"Precision: {precision_overall.avg:.4f}, Recall: {recall_overall.avg:.4f}, F1: {f1_overall.avg:.4f}, \n"
        f"Model fps rate: {fps:.0f}"
    )
    logger.info(f"PCK Results: {pck_results}")


if __name__ == '__main__':
    main()
