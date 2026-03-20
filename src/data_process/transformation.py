import random

import cv2
import numpy as np
import torchvision.transforms.functional as F
import torch


class Compose(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, imgs, ball_position_xy, visibility):
        if random.random() <= self.p:
            for t in self.transforms:
                result = t(imgs, ball_position_xy, visibility)
                imgs, ball_position_xy, visibility = t(imgs, ball_position_xy, visibility)
        return imgs, ball_position_xy, visibility


class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), num_frames_sequence=9, p=1.0):
        self.p = p
        self.mean = np.array(mean).reshape(1, 1, 3)  # For individual image normalization
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, imgs, ball_position_xy, visibility):
        if random.random() < self.p:
            imgs = [((img / 255.) - self.mean) / self.std for img in imgs]

        return imgs, ball_position_xy, visibility


class Denormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0):
        self.p = p
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, img):
        img = (img * self.std + self.mean) * 255.
        img = img.astype(np.uint8)

        return img


class Resize(object):
    def __init__(self, new_size, p=0.5, interpolation=cv2.INTER_LANCZOS4):
        self.new_size = new_size  # new_size should be (height, width)
        self.p = p
        self.interpolation = interpolation

    def __call__(self, imgs, ball_position_xy, visibility):
        """_summary_

        Args:
            imgs (numpy): list of images
            ball_position_xy (numpy): ball position of intended frame 

        Returns:
            _type_: _description_
        """
        # If random value is greater than p, return original imgs and ball position
        if random.random() > self.p:
            return imgs, ball_position_xy, visibility

        # Original image dimensions (assuming imgs[0] has the original size)
        original_h, original_w, _ = imgs[0].shape

        # New image dimensions
        new_h, new_w = self.new_size
        
        # Resize a sequence of images
        transformed_imgs = []
        for img in imgs:
            transformed_img = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
            transformed_imgs.append(transformed_img)

        # Adjust ball position
        w_ratio = float(original_w) / float(new_w)  # New width divided by original width
        h_ratio = float(original_h) / float(new_h) # New height divided by original height

        transformed_ball_pos = np.array([ball_position_xy[0] / w_ratio, ball_position_xy[1] / h_ratio])
        # Round the coordinates to the nearest integer
        # transformed_ball_pos = np.round(transformed_ball_pos).astype(int)
        
        return transformed_imgs, transformed_ball_pos, visibility


class Random_Crop(object):
    def __init__(self, max_reduction_percent=0.15, p=0.5, interpolation=cv2.INTER_LINEAR):
        self.max_reduction_percent = max_reduction_percent
        self.p = p
        self.interpolation = interpolation

    def __call__(self, imgs, ball_position_xy, visibility):
        transformed_imgs = imgs.copy()
        transformed_ball_pos = ball_position_xy.copy()
        # imgs are before resizing
        if random.random() <= self.p:
            h, w, c = imgs[0].shape
            # Calculate min_x, max_x, min_y, max_y
            remain_percent = random.uniform(1. - self.max_reduction_percent, 1.)
            new_w = remain_percent * w
            min_x = int(random.uniform(0, w - new_w))
            max_x = int(min_x + new_w)
            w_ratio = w / new_w

            new_h = remain_percent * h
            min_y = int(random.uniform(0, h - new_h))
            max_y = int(new_h + min_y)
            h_ratio = h / new_h
            # crop a sequence of images
            transformed_imgs = []
            for i, img in enumerate(imgs):
                img_cropped = img[min_y:max_y, min_x:max_x, :]
                # Resize the image to the original dimensions
                img_resized = cv2.resize(img_cropped, (w, h), interpolation=self.interpolation)
                transformed_imgs.append(img_resized)
                if i == len(imgs):
                    transformed_ball_pos = np.array([(ball_position_xy[0] - min_x) * w_ratio,
                                            (ball_position_xy[1] - min_y) * h_ratio])

        return transformed_imgs, transformed_ball_pos, visibility
    
class Center_Crop(object):
    def __init__(self, target_size=(224, 224), p=1.0):
        """
        Args:
            target_size (tuple): The (height, width) to center crop to.
            p (float): Probability of applying the transform.
        """
        self.target_h, self.target_w = target_size
        self.p = p

    def __call__(self, imgs, ball_position_xy, visibility):
        transformed_imgs = imgs.copy()
        transformed_ball_pos = ball_position_xy.copy()

        if random.random() <= self.p:
            h, w, c = imgs[0].shape

            # Ensure target size doesn't exceed original size
            assert self.target_h <= h and self.target_w <= w, "Target crop size must be less than or equal to original size"

            # Compute crop window centered
            min_x = (w - self.target_w) // 2
            max_x = min_x + self.target_w
            min_y = (h - self.target_h) // 2
            max_y = min_y + self.target_h

            # Apply center crop
            transformed_imgs = [img[min_y:max_y, min_x:max_x, :] for img in imgs]

            # Update ball position
            transformed_ball_pos = np.array([
                ball_position_xy[0] - min_x,
                ball_position_xy[1] - min_y
            ])

        return transformed_imgs, transformed_ball_pos, visibility


class Random_Rotate(object):
    def __init__(self, rotation_angle_limit=15, p=0.5):
        self.rotation_angle_limit = rotation_angle_limit
        self.p = p

    def __call__(self, imgs, ball_position_xy, visibility):
        transformed_imgs = imgs.copy()
        if random.random() <= self.p:
            random_angle = random.uniform(-self.rotation_angle_limit, self.rotation_angle_limit)
            # Rotate a sequence of imgs
            h, w, c = imgs[0].shape
            center = (int(w / 2), int(h / 2))
            rotate_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.)

            transformed_imgs = []
            for img in imgs:
                transformed_img = cv2.warpAffine(img, rotate_matrix, (w, h), flags=cv2.INTER_LINEAR)
                transformed_imgs.append(transformed_img)

            # Adjust ball position, apply the same rotate_matrix for the sequential images
            ball_position_xy = rotate_matrix.dot(np.array([ball_position_xy[0], ball_position_xy[1], 1.]).T)

        return transformed_imgs, ball_position_xy, visibility


class Random_HFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, ball_position_xy, visibility):
        transformed_imgs = imgs.copy()
        transformed_ball_position_xy = ball_position_xy.copy()
        if random.random() <= self.p:
            h, w, c = imgs[0].shape
            transformed_imgs = []
            for img in imgs:
                # Horizontal flip a sequence of imgs
                transformed_img = cv2.flip(img, 1)
                transformed_imgs.append(transformed_img)

            # Adjust ball position: Same y, new x = w - x
            transformed_ball_position_xy[0] = w - transformed_ball_position_xy[0]
       
        return transformed_imgs, transformed_ball_position_xy, visibility


class Random_VFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, ball_position_xy, visibility):
        transformed_imgs = imgs.copy()
        if random.random() <= self.p:
            h, w, c = imgs[0].shape
            transformed_imgs = []
            for img in imgs:
                # Horizontal flip a sequence of imgs
                transformed_img = cv2.flip(img, 0)
                transformed_imgs.append(transformed_img)

            # Adjust ball position: Same x, new y = h - y
            ball_position_xy[1] = h - ball_position_xy[1]

        return transformed_imgs, ball_position_xy, visibility



class Random_Ball_Mask:
    def __init__(self, target_frame, mask_size=(20, 20), p=0.5, mask_type='mean', shapes=['rectangle', 'circle', 'ellipse']):
        """
        Args:
            mask_size (tuple): Height and width of the mask area (blackout area).
            p (float): Probability of applying the mask.
            mask_type (str): Type of mask ('zero', 'noise', 'mean').
            shapes (list of str): List of shapes to choose from ('rectangle', 'circle', 'ellipse').
        """
        self.target_frame = target_frame
        self.mask_size = mask_size
        self.p = p
        self.mask_type = mask_type
        self.shapes = shapes

    def __call__(self, imgs, ball_position_xy, visibility):
        """
        Args:
            imgs : List of numpy arrays where the length represents num frames
            ball_position_xy (numpy): (x, y) ball position for the labeled frame.

        Returns:
            masked_imgs: Tensor with the ball area masked in some frames.
        """
        H, W, C = imgs[0].shape  # Get the shape from the input tensor
        movement_range = self.mask_size[0] // 2
        mask_h = random.randint(self.mask_size[0] - movement_range, self.mask_size[0] + movement_range)
        mask_w = random.randint(self.mask_size[1] - movement_range, self.mask_size[1] + movement_range)

        for i, img in enumerate(imgs):
            if random.random() < self.p:
                if i == self.target_frame:
                    x, y = int(ball_position_xy[0]), int(ball_position_xy[1])
                    visibility = 3
                else:
                    # Apply mask at a random position in non-labeled frames
                    x = random.randint(0, W - mask_w)
                    y = random.randint(0, H - mask_h)

                # Ensure the mask is within the image boundaries
                top = max(0, min(H - mask_h, y - mask_h // 2))
                left = max(0, min(W - mask_w, x - mask_w // 2))

                # Randomly select a shape
                shape = random.choice(self.shapes)

                # Create the mask
                mask = np.zeros_like(img, dtype=np.uint8)

                if shape == 'rectangle':
                    cv2.rectangle(mask, (left, top), (left + mask_w, top + mask_h), (1, 1, 1), -1)

                elif shape == 'circle':
                    radius = min(mask_h, mask_w) // 2
                    center = (x, y)
                    cv2.circle(mask, center, radius, (1, 1, 1), -1)

                elif shape == 'ellipse':
                    center = (x, y)
                    axes = (mask_w // 2, mask_h // 2)
                    angle = random.randint(0, 360)  # Random rotation
                    cv2.ellipse(mask, center, axes, angle, 0, 360, (1, 1, 1), -1)

                # Generate the masked region based on the mask type
                if self.mask_type == 'zero':
                    img[mask.astype(bool)] = 0

                elif self.mask_type == 'noise':
                    noise = np.random.randn(*img.shape) * 255  # Generate noise in the same shape
                    img[mask.astype(bool)] = noise.clip(0, 255)[mask.astype(bool)]

                elif self.mask_type == 'mean':
                    mean_value = img[mask.astype(bool)].mean(axis=0) if mask.astype(bool).sum() > 0 else 0
                    noise = np.random.randn(*img.shape) * 10  # Add small noise
                    img[mask.astype(bool)] = (mean_value + noise.clip(0, 255))[mask.astype(bool)]

        return imgs, ball_position_xy, visibility



class RandomColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5):
        """
        Initializes the RandomColorJitter augmentation.

        Args:
            brightness (float): Brightness adjustment factor (default 0.2).
            contrast (float): Contrast adjustment factor (default 0.2).
            saturation (float): Saturation adjustment factor (default 0.1).
            hue (float): Hue shift factor (default 0.1).
            p (float): Probability of applying the jitter (default 0.5).
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, imgs, ball_position_xy, visibility):
        """
        Applies random color jitter to a sequence of images.

        Args:
            imgs (list of np.ndarray): List of images to augment.

        Returns:
            list of np.ndarray: Augmented images.
        """
        transformed_imgs = imgs.copy()

        # Apply jitter with probability p
        if random.random() <= self.p:
            transformed_imgs = []
            for img in imgs:
                jittered_img = self.apply_jitter(img)
                transformed_imgs.append(jittered_img)

        return transformed_imgs, ball_position_xy, visibility

    def apply_jitter(self, img):
        """
        Applies brightness, contrast, saturation, and hue jitter to a single image.

        Args:
            img (np.ndarray): Input image (H, W, C) in range [0, 255].

        Returns:
            np.ndarray: Jittered image.
        """
        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Apply brightness jitter
        brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        img = np.clip(img * brightness_factor, 0, 1)

        # Apply contrast jitter
        contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * contrast_factor + mean, 0, 1)

        # Convert to HSV to apply saturation and hue jitter
        hsv_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

        # Apply saturation jitter
        saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
        hsv_img[..., 1] = np.clip(hsv_img[..., 1] * saturation_factor, 0, 255)

        # Apply hue jitter
        hue_shift = np.random.uniform(-self.hue * 180, self.hue * 180)
        hsv_img[..., 0] = (hsv_img[..., 0] + hue_shift) % 180

        # Convert back to RGB
        jittered_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        # Convert back to uint8 and scale to [0, 255]
        jittered_img = np.clip(jittered_img * 255, 0, 255).astype(np.uint8)

        return jittered_img


