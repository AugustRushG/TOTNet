"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.10
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: This script creates the video loader for testing with an input video
"""

import os
from collections import deque

import cv2
import numpy as np
import sys
sys.path.append('../')


class Video_Loader:
    """The loader for demo with a video input."""

    def __init__(self, video_path, input_size=(288, 512), num_frames=5, 
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Args:
            video_path (str): Path to the video file.
            input_size (tuple): Desired input size (height, width).
            num_frames (int): Number of frames per sequence.
            mean (tuple): Mean values for normalization.
            std (tuple): Standard deviation values for normalization.
        """
        assert os.path.isfile(video_path), f"No video at {video_path}"
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = input_size[1]
        self.height = input_size[0]
        self.count = 0
        self.num_frames_sequence = num_frames
        self.mean = np.array(mean).reshape(1, 1, 3)  # Reshape for broadcasting
        self.std = np.array(std).reshape(1, 1, 3)    # Reshape for broadcasting

        print(f'Length of the video: {self.video_num_frames} frames')

        self.images_sequence = deque(maxlen=num_frames)
        self.get_first_images_sequence()

    def normalize(self, img):
        """Normalize an individual frame."""
        img = img.astype(np.float32) / 255.0  # Scale to [0, 1]
        return (img - self.mean) / self.std  # Normalize using mean and std

    def get_first_images_sequence(self):
        """Load the first `num_frames_sequence` frames and normalize them."""
        while self.count < self.num_frames_sequence:
            self.count += 1
            ret, frame = self.cap.read()  # BGR
            assert ret, f'Failed to load frame {self.count}'
            
            # Resize, convert to RGB, and normalize
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height))
            normalized_frame = self.normalize(frame)
            self.images_sequence.append(normalized_frame)

    def __iter__(self):
        self.count = self.num_frames_sequence - 1  # Start from the first available sequence
        return self

    def __next__(self):
        self.count += 1
        if self.count >= self.video_num_frames:
            raise StopIteration

        # Read the next frame
        ret, frame = self.cap.read()  # BGR
        assert ret, f'Failed to load frame {self.count}'

        # Resize, convert to RGB, and normalize
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height))
        normalized_frame = self.normalize(frame)
        self.images_sequence.append(normalized_frame)

        # Prepare images in [N, C, H, W] format
        frames_np = np.array(self.images_sequence)  # [N, H, W, C]
        frames_np = frames_np.transpose(0, 3, 1, 2)  # [N, C, H, W]

        return self.count, frames_np, frame, original_frame

    def __len__(self):
        return self.video_num_frames - self.num_frames_sequence  # Number of sequences
    



class Video_Loader_MIMO:
    """The loader for demo with a video input."""

    def __init__(self, video_path, input_size=(288, 512), num_frames=5, 
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Args:
            video_path (str): Path to the video file.
            input_size (tuple): Desired input size (height, width).
            num_frames (int): Number of frames per sequence.
            mean (tuple): Mean values for normalization.
            std (tuple): Standard deviation values for normalization.
        """
        assert os.path.isfile(video_path), f"No video at {video_path}"
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = input_size[1]
        self.height = input_size[0]
        self.count = 0
        self.num_frames_sequence = num_frames
        self.mean = np.array(mean).reshape(1, 1, 3)  # Reshape for broadcasting
        self.std = np.array(std).reshape(1, 1, 3)    # Reshape for broadcasting

        print(f'Length of the video: {self.video_num_frames} frames')

        self.images_sequence = deque(maxlen=num_frames)
        self.get_first_images_sequence()

    def normalize(self, img):
        """Normalize an individual frame."""
        img = img.astype(np.float32) / 255.0  # Scale to [0, 1]
        return (img - self.mean) / self.std  # Normalize using mean and std

    def get_first_images_sequence(self):
        """Load the first `num_frames_sequence` frames and normalize them."""
        while self.count < self.num_frames_sequence:
            self.count += 1
            ret, frame = self.cap.read()  # BGR
            assert ret, f'Failed to load frame {self.count}'
            
            # Resize, convert to RGB, and normalize
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height))
            normalized_frame = self.normalize(frame)
            self.images_sequence.append(normalized_frame)

    def __iter__(self):
        self.count = self.num_frames_sequence - 1  # Start from the first available sequence
        return self

    def __next__(self):
        self.count += self.num_frames_sequence
        if self.count >= self.video_num_frames:
            raise StopIteration
        
        self.images_sequence.clear()

        for i in range(self.num_frames_sequence):
            # Read the next frame
            ret, frame = self.cap.read()  # BGR
            assert ret, f'Failed to load frame {self.count}'
            
            # Resize, convert to RGB, and normalize
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height))
            normalized_frame = self.normalize(frame)
            self.images_sequence.append(normalized_frame)

        # Prepare images in [N, C, H, W] format
        frames_np = np.array(self.images_sequence)  # [N, H, W, C]
        frames_np = frames_np.transpose(0, 3, 1, 2)  # [N, C, H, W]

        return self.count, frames_np, frame

    def __len__(self):
        return self.video_num_frames - self.num_frames_sequence  # Number of sequences
    



if __name__ == '__main__':
    import time

    import matplotlib.pyplot as plt
    from config.config import parse_configs

    configs = parse_configs()
    configs.num_frames = 5

    video_path = os.path.join(configs.dataset_dir, 'test', 'videos', 'test_1.mp4')
    print(video_path)
    video_loader = Video_Loader(video_path, input_size=(512, 288),
                                      num_frames=configs.num_frames)
    out_images_dir = os.path.join(configs.results_dir, 'debug', 'ttnet_video_loader')
    if not os.path.isdir(out_images_dir):
        os.makedirs(out_images_dir)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    axes = axes.ravel()

    for example_index in range(1, 10):
        print('process the sequence index: {}'.format(example_index))
        start_time = time.time()
        count, resized_imgs = video_loader.__next__()
        print(resized_imgs.shape)
        print('time to load sequence {}: {}'.format(example_index, time.time() - start_time))

        resized_imgs = resized_imgs.transpose(1, 2, 0)
        for i in range(configs.num_frames):
            img = resized_imgs[:, :, (i * 3): (i + 1) * 3]
            axes[i].imshow(img)
            axes[i].set_title('image {}'.format(i))
        plt.savefig(os.path.join(out_images_dir, 'augment_all_imgs_{}.jpg'.format(example_index)))

        origin_imgs = cv2.resize(resized_imgs, (1920, 1080))
        for i in range(configs.num_frames):
            img = origin_imgs[:, :, (i * 3): (i + 1) * 3]
            axes[i].imshow(img)
            axes[i].set_title('image {}'.format(i))
        plt.savefig(os.path.join(out_images_dir, 'org_all_imgs_{}.jpg'.format(example_index)))
