import sys

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader, Subset

sys.path.append('../')

from data_process.dataset import Occlusion_Dataset, Tennis_Dataset, Event_Dataset, Badminton_Dataset, TTA_Dataset
from data_process.transformation import Compose, Resize, Normalize, Random_Rotate, Random_HFlip, Random_VFlip, Random_Ball_Mask, RandomColorJitter
from data_process.data_utils import (
    get_all_detection_infor,
    train_val_data_separation,
    get_all_detection_infor_tennis,
    get_events_infor_noseg,
    get_all_detection_infor_badminton,
    get_new_tracking_infor
)



def create_occlusion_train_val_dataloader(configs, subset_size=None, necessary_prob=1.0):
    """Create dataloader for training and validation, with an option to use a subset of the data."""
    target_frame = configs.num_frames//2 if configs.bidirect else configs.num_frames-1
    train_transform = Compose([
        RandomColorJitter(p=0.3),
        Random_Ball_Mask(target_frame=target_frame,mask_size=(configs.img_size[0]//10,configs.img_size[0]//10), p=configs.occluded_prob),
        Random_HFlip(p=0.2),
        Random_VFlip(p=0),
        Random_Rotate(rotation_angle_limit=5, p=0.1),
        Resize(new_size=configs.img_size, p=necessary_prob),
        Normalize(num_frames_sequence=configs.num_frames, p=necessary_prob),
    ], p=1.)

    print("No normalization in occlusion dataloader")

 
    # Load train and validation data information
    train_events_infor, val_events_infor, train_events_label, val_events_label = train_val_data_separation(configs)

    if configs.dataset_choice == 'tt':
        train_dataset = Occlusion_Dataset(train_events_infor, train_events_label, transform=train_transform,
                                        num_samples=configs.num_samples)
    elif configs.dataset_choice == 'tennis':
        train_dataset = Tennis_Dataset(train_events_infor, train_events_label, transform=train_transform,
                                    num_samples=configs.num_samples)
    elif configs.dataset_choice == 'badminton':
        train_dataset = Badminton_Dataset(train_events_infor, train_events_label, transform=train_transform,
                                    num_samples=configs.num_samples)
    elif configs.dataset_choice == 'tta':
        train_dataset = TTA_Dataset(train_events_infor, train_events_label, transform=train_transform,
                                    num_samples=configs.num_samples)
    # If subset_size is provided, create a subset for training
    if subset_size is not None:
        train_indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        train_dataset = Subset(train_dataset, train_indices)
    
    # Create train sampler if distributed
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    # Create train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, 
                                  sampler=train_sampler, drop_last=False)

    # Create validation dataloader (without transformations)
    val_dataloader = None
    if not configs.no_val:
        val_transform = Compose([
            Resize(new_size=configs.img_size, p=necessary_prob),
            # Center_Crop(target_size=(224,224), p=necessary_prob),
            Normalize(num_frames_sequence=configs.num_frames, p=necessary_prob),
        ], p=1.)

        if configs.dataset_choice == 'tt':
            val_dataset = Occlusion_Dataset(val_events_infor, val_events_label, transform=val_transform,
                                            num_samples=configs.num_samples)
        elif configs.dataset_choice == 'tennis':
            val_dataset = Tennis_Dataset(val_events_infor, val_events_label, transform=val_transform,
                                        num_samples=configs.num_samples)
        elif configs.dataset_choice == 'badminton':
            val_dataset = Badminton_Dataset(val_events_infor, val_events_label, transform=val_transform,
                                        num_samples=configs.num_samples)
        elif configs.dataset_choice == 'tta':
            val_dataset = TTA_Dataset(val_events_infor, val_events_label, transform=val_transform,
                                        num_samples=configs.num_samples)

        # If subset_size is provided, create a subset for validation
        if subset_size is not None:
            val_indices = torch.randperm(len(val_dataset))[:subset_size].tolist()
            val_dataset = Subset(val_dataset, val_indices)
        
        # Create validation sampler if distributed
        val_sampler = None
        if configs.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        
        # Create validation dataloader
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                    pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler, drop_last=False)

    return train_dataloader, val_dataloader, train_sampler

def create_occlusion_test_dataloader(configs, subset_size=None):
    """Create dataloader for testing phase"""

    test_transform = Compose([
            Resize(new_size=configs.img_size, p=1.0),
            # Resize(new_size=(288, 512), p=1.0),
            # Center_Crop(target_size=(224,224), p=1.0),
            Normalize(num_frames_sequence=configs.num_frames, p=1),
        ], p=1.)
    dataset_type = 'test'
    if configs.dataset_choice == 'tt':
        if configs.event:
            test_events_infor, test_events_labels = get_events_infor_noseg(configs.test_game_list, configs, dataset_type)
            test_dataset = Event_Dataset(test_events_infor, test_events_labels, transform=test_transform,
                                 num_samples=configs.num_samples)
        else:
            test_events_infor, test_events_labels = get_all_detection_infor(configs.test_game_list, configs, dataset_type)
            test_dataset = Occlusion_Dataset(test_events_infor, test_events_labels, transform=test_transform,
                                    num_samples=configs.num_samples)
    elif configs.dataset_choice == 'tennis':
        test_events_infor, test_events_labels = get_all_detection_infor_tennis(configs.tennis_test_game_list, configs)
        test_dataset = Tennis_Dataset(test_events_infor, test_events_labels, transform=test_transform,
                                 num_samples=configs.num_samples)
    elif configs.dataset_choice == 'badminton':
        test_events_infor, test_events_labels = get_all_detection_infor_badminton(configs.badminton_test_game_list, configs)
        test_dataset = Badminton_Dataset(test_events_infor, test_events_labels, transform=test_transform,
                                 num_samples=configs.num_samples)
    elif configs.dataset_choice == 'tta':
        test_events_infor, test_events_labels = get_new_tracking_infor(configs.tta_tracking_dataset_dir, 'test', num_frames=configs.num_frames, resize=configs.resize, bidirect=configs.bidirect)
        test_dataset = TTA_Dataset(test_events_infor, test_events_labels, transform=test_transform,
                                num_samples=configs.num_samples)
    test_sampler = None

    # If subset_size is provided, create a subset for training
    if subset_size is not None:
        test_indices = torch.randperm(len(test_dataset))[:subset_size].tolist()
        test_dataset = Subset(test_dataset, test_indices)

    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, 
                                 sampler=test_sampler, drop_last=False)

    return test_dataloader



def draw_image_with_ball(image_tensor, ball_location_tensor, out_images_dir, example_index):
    # Convert tensors to numpy arrays

    image = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    ball_location = ball_location_tensor.cpu().numpy()    # Ensure ball location is on CPU

    # Ensure the image is in uint8 format for OpenCV
    if image.dtype != 'uint8':
        image = (image * 255).astype('uint8')

    # Draw the ball on the image
    ball_xy = tuple(ball_location.astype(int))  # Convert coordinates to int
    img_with_ball = cv2.circle(image.copy(), ball_xy, radius=5, color=(255, 0, 0), thickness=2)

    # Convert the image to BGR format for saving with OpenCV
    img_with_ball = cv2.cvtColor(img_with_ball, cv2.COLOR_RGB2BGR)

    # Save the image
    output_path = os.path.join(out_images_dir, f'example_label_{example_index}.jpg')
    cv2.imwrite(output_path, img_with_ball)

    return output_path  # Optionally return the saved path

def concatenate_images_horizontally(images):
    """
    Concatenate a list of images horizontally.
    Args:
        images (list): List of images (numpy arrays) to concatenate.
    Returns:
        concatenated_image: Combined image as a single numpy array.
    """
    return cv2.hconcat(images)

if __name__ == '__main__':
    from config.config import parse_configs
    import random

    configs = parse_configs()
    configs.distributed = False  # For testing
    configs.batch_size = 1
    # configs.img_size = (1080, 1920)
    configs.img_size = (288,512)
    configs.interval = 1
    configs.num_frames = 5
    configs.occluded_prob = 0
    configs.bidirect = True
    configs.dataset_choice = 'tta'
    configs.mimo = False
    # configs.new_data = True
    # configs.event = True
    # configs.smooth_labelling = True
    
    seed = np.random.randint(0, 2**32)  # Generate a random integer seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create Masked dataloaders 
    train_dataloader, val_dataloader, train_sampler = create_occlusion_train_val_dataloader(configs, necessary_prob=1.0)
    test_dataloader = create_occlusion_test_dataloader(configs, configs.num_samples)

    print('len train_dataloader: {}, val_dataloader: {}'.format(len(train_dataloader), len(val_dataloader)))
    print(f"len test_loader {len(test_dataloader)}")

    frame_id = configs.num_frames-1

    batch_data, (masked_frameids, ball_xys, visibility, events) = next(iter(test_dataloader))
    # last_batch = list(train_dataloader)[0]
    # batch_data, (masked_frameids, ball_xys, visibility, events) = last_batch

    if configs.event or configs.bidirect:
        frame_id = configs.num_frames//2

    print(f"frame id is {frame_id}")
    
    # print(f"unique number of batch_data is {torch.unique(batch_data)}")
    # Check the shapes
    # print(f'ball frame is {frame_id}, print event is {event_classes}')
    print(f'Batch data shape: {batch_data.shape}')      # Expected: [B, N, C, H, W]

    if configs.mimo:
        print(ball_xys)
        print(f'Batch ball_xy length {len(ball_xys)} shape: {ball_xys[0].shape}')  # Expected: [8, 2], 2 represents X and Y of the coordinaties 
    else:
        print(f'Batch ball_xy shape: {ball_xys.shape}')  # Expected: [8, 2], 2 represents X and Y of the coordinaties 
        print(torch.unique(ball_xys))
   
   
    # Select the first sample in the batch
    sample_data = batch_data[0]  # Shape: [B, C, H, W]

    # Select the first frame
    img = sample_data[frame_id]  # Shape: [C, H, W]

    # Transpose the dimensions to [H, W, C]
    image = np.transpose(img, (1, 2, 0))  # Shape: [H, W, C]
    image = image.cpu().numpy()
    
    out_images_dir = os.path.join(configs.results_dir, 'debug', 'ttnet_dataset')
    if not os.path.isdir(out_images_dir):
        os.makedirs(out_images_dir)
    cv2.imwrite(os.path.join(out_images_dir, f'example.jpg'), image)

    # Loop over each sample in the batch
    for batch_index in range(batch_data.shape[0]):
        sample_data = batch_data[batch_index]  # Shape: [N, C, H, W]
        masked_image = sample_data[frame_id]  # Shape: [C, H, W]
        ball_xy = ball_xys[batch_index].cpu().numpy()  # Ball coordinates for this sample, as a list
        if configs.mimo:
            ball_xy = (int(ball_xy[frame_id][0]), int(ball_xy[frame_id][1]))
            print(ball_xy)
        # Collect all frames (original and masked) for visualization
        frame_images = []

        for frame_index in range(sample_data.shape[0]):
            img = sample_data[frame_index]  # Shape: [C, H, W]
            image = np.transpose(img.cpu().numpy(), (1, 2, 0))  # Convert to [H, W, C]
            frame_images.append(image)

        # Add the masked frame with ball position
        masked_frame = np.transpose(masked_image.cpu().numpy(), (1, 2, 0))  # Convert to [H, W, C]
        img_with_ball = cv2.circle(masked_frame.copy(), tuple(ball_xy), radius=10, color=(0, 0, 255), thickness=3)
        # img_with_ball = cv2.cvtColor(img_with_ball, cv2.COLOR_RGBR)  # Convert to BGR for saving
        frame_images.append(img_with_ball)  # Add the masked frame to the list

        # Concatenate all frames horizontally
        combined_image = concatenate_images_horizontally(frame_images)

        # Save the combined image
        output_path = os.path.join(out_images_dir, f'test_batch_{batch_index}_combined.jpg')
        cv2.imwrite(output_path, combined_image)
        output_path = os.path.join(out_images_dir, f'test_batch_{batch_index}_masked.jpg')
        cv2.imwrite(output_path, img_with_ball)

    print("All combined images saved successfully.")