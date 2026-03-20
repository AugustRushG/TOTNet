from typing import Union, Optional
import torch
import numpy as np
import csv
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch


def extract_coords2d(pred_heatmap, H, W):
    """
    Extracts (x, y) coordinates from a predicted flattened 2D heatmap.

    Args:
        pred_heatmap (Tensor):
            Predicted heatmap of shape [B, H*W].
            Can be probabilities or logits.
        H (int): height of the heatmap.
        W (int): width of the heatmap.

    Returns:
        Tensor: [B, 2] with (x, y) coordinates for each batch item.
    """
    if pred_heatmap.dim() != 2 or pred_heatmap.size(1) != H * W:
        raise ValueError(f"Expected shape [B, {H*W}], got {list(pred_heatmap.shape)}")

    B = pred_heatmap.size(0)

    # Argmax to get the flat index of the highest-probability pixel
    flat_idx = pred_heatmap.argmax(dim=1)  # [B]

    # Convert flat index to (x, y)
    x_pred = (flat_idx % W).float()        # [B]
    y_pred = (flat_idx // W).float()       # [B]

    # Stack into [B, 2]
    pred_coords = torch.stack([x_pred, y_pred], dim=1)

    return pred_coords


def extract_coords(pred_heatmap):
    """_summary_

    Args:
        pred_heatmap : tuple of tensors (pred_x_logits, pred_y_logits)
        - pred_x_logits: Tensor of shape [B, W] with predicted logits for x-axis
        - pred_y_logits: Tensor of shape [B, H] with predicted logits for y-axis
    Return:
        out (tensor) : Tensor in shape [B,2] which represents coords for each 
    """
    pred_x_logits, pred_y_logits = pred_heatmap

    # Predicted coordinates are extracted by taking the argmax over logits
    x_pred_indices = torch.argmax(pred_x_logits, dim=1)  # [B]
    y_pred_indices = torch.argmax(pred_y_logits, dim=1)  # [B]

    # Convert indices to float for calculations
    x_pred = x_pred_indices.float()
    y_pred = y_pred_indices.float()

    # Stack the predicted x and y coordinates
    pred_coords = torch.stack([x_pred, y_pred], dim=1)  # [B, 2]

    return pred_coords


def extract_coords_mimo(pred_heatmap):
    """
    Args:
        pred_heatmap: tuple of tensors (pred_x_logits, pred_y_logits)
            - pred_x_logits: [B, N, W]
            - pred_y_logits: [B, N, H]
              where B = batch size, N = number of frames, W/H = width/height logits
    Returns:
        pred_coords: [B, N, 2] containing (x, y) coordinates for each frame
    """
    pred_x_logits, pred_y_logits = pred_heatmap
    # pred_x_logits: [B, N, W]
    # pred_y_logits: [B, N, H]

    # For each sample in B, each frame in N, take argmax along width/height dimension
    x_pred_indices = torch.argmax(pred_x_logits, dim=2)  # [B, N]
    y_pred_indices = torch.argmax(pred_y_logits, dim=2)  # [B, N]

    # Convert indices to float
    x_pred = x_pred_indices.float()
    y_pred = y_pred_indices.float()

    # Stack along the last dimension => [B, N, 2]
    pred_coords = torch.stack([x_pred, y_pred], dim=2)

    return pred_coords



def heatmap2d_calculate_metrics(pred_map, target_coords, H, W, scale=None):
    """
    Calculates metrics between a predicted flattened 2D heatmap and target (x,y) coords.

    Args:
    - pred_map: Tensor [B, H*W] with logits/probabilities over pixels
    - target_coords: Tensor [B,2] with ground-truth (x, y) coords
    - H, W: int, height and width of the heatmap
    - scale: Optional scaling for predicted coords:
        * scalar (float/int)
        * Tensor [] (scalar), [2] (sx,sy), [B] (per-sample scalar), or [B,2] (per-sample (sx,sy))

    Returns:
    - mse, rmse, mae, euclidean_distance (Python floats)
    """
    if pred_map.dim() != 2 or pred_map.size(1) != H * W:
        raise ValueError(f"pred_map must be [B, {H*W}], got {list(pred_map.shape)}")

    B = pred_map.size(0)
    device = pred_map.device

    target_coords = target_coords.to(device).float()

    # Argmax over flattened spatial dimension
    flat_idx = pred_map.argmax(dim=1)          # [B]
    x_pred = (flat_idx % W).float()            # [B]
    y_pred = (flat_idx // W).float()           # [B]
    pred_coords = torch.stack([x_pred, y_pred], dim=1)  # [B,2]

    # Optional scaling
    if scale is not None:
        if torch.is_tensor(scale):
            s = scale.to(device).float()
            if s.ndim == 0:                        # scalar
                pred_coords = pred_coords * s
            elif s.ndim == 1:
                if s.numel() == 2:                 # (sx, sy)
                    pred_coords = pred_coords * s.view(1, 2)
                elif s.numel() == B:               # per-sample scalar
                    pred_coords = pred_coords * s.view(B, 1)
                else:
                    raise ValueError("scale 1D tensor must be length 2 or B")
            elif s.ndim == 2 and s.shape == (B, 2):  # per-sample (sx, sy)
                pred_coords = pred_coords * s
            else:
                raise ValueError("scale shape must be [], [2], [B], or [B,2]")
        else:
            pred_coords = pred_coords * float(scale)

    # Compute errors
    diff = pred_coords - target_coords  # [B,2]

    mse_per_sample = (diff ** 2).mean(dim=1)   # [B]
    mse = mse_per_sample.mean()

    rmse_per_sample = torch.sqrt(mse_per_sample + 1e-12)  # [B]
    rmse = rmse_per_sample.mean()

    mae_per_sample = diff.abs().mean(dim=1)    # [B]
    mae = mae_per_sample.mean()

    euclidean_distance_per_sample = torch.norm(diff, dim=1)  # [B]
    euclidean_distance = euclidean_distance_per_sample.mean()

    return mse.item(), rmse.item(), mae.item(), euclidean_distance.item()


def heatmap_calculate_metrics(pred_logits, target_coords, scale=None):
    """
    Calculates evaluation metrics between predicted logits for x and y coordinates
    and target pixel coordinates (integers).
    
    Args:
    - pred_logits: tuple of tensors (pred_x_logits, pred_y_logits)
        - pred_x_logits: Tensor of shape [B, W] with predicted logits for x-axis
        - pred_y_logits: Tensor of shape [B, H] with predicted logits for y-axis
    - target_coords: Tensor of shape [B, 2] with ground truth (x, y) coordinates (integers)
    - scale: Tensor
    
    Returns:
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
    - euclidean_distance: Average Euclidean distance between predictions and ground truth
    """
    pred_x_logits, pred_y_logits = pred_logits

    # Ensure tensors are on the same device
    device = pred_x_logits.device
    target_coords = target_coords.to(device)

    # Predicted coordinates are extracted by taking the argmax over logits
    x_pred_indices = torch.argmax(pred_x_logits, dim=1)  # [B]
    y_pred_indices = torch.argmax(pred_y_logits, dim=1)  # [B]

    # Convert indices to float for calculations
    x_pred = x_pred_indices.float()
    y_pred = y_pred_indices.float()

    # Stack the predicted x and y coordinates
    pred_coords = torch.stack([x_pred, y_pred], dim=1)  # [B, 2]

    if scale is not None:
        pred_coords = pred_coords * scale

    # Convert target coordinates to float
    target_coords = target_coords.float()

    # Difference between predicted and ground truth coordinates
    diff = pred_coords - target_coords  # [B, 2]

    # Mean Squared Error (MSE) per sample
    mse_per_sample = torch.mean(diff ** 2, dim=1)  # [B]

    # Mean MSE over all samples
    mse = mse_per_sample.mean()

    # Root Mean Squared Error (RMSE) per sample
    rmse_per_sample = torch.sqrt(mse_per_sample)  # [B]

    # Mean RMSE over all samples
    rmse = rmse_per_sample.mean()

    # Mean Absolute Error (MAE) per sample
    mae_per_sample = torch.mean(torch.abs(diff), dim=1)  # [B]

    # Mean MAE over all samples
    mae = mae_per_sample.mean()

    # Euclidean distance per sample
    euclidean_distance_per_sample = torch.norm(diff, dim=1)  # [B]

    # Mean Euclidean distance over all samples
    euclidean_distance = euclidean_distance_per_sample.mean()

    return mse.item(), rmse.item(), mae.item(), euclidean_distance.item()



def heatmap_calculate_metrics_multi(
    pred_logits,
    target_coords,
    scale=None
):
    """
    Calculates evaluation metrics (MSE, RMSE, MAE, Euclidean distance)
    for multi-frame predicted logits vs. ground-truth coordinates, skipping
    frames where target_coords == (0,0).

    Args:
        pred_logits: tuple (pred_x_logits, pred_y_logits)
          - pred_x_logits: [B, N, W]  (B=batch, N=frames, W=width)
          - pred_y_logits: [B, N, H]  (B=batch, N=frames, H=height)
        target_coords: [B, N, 2] ground-truth (x, y) integer pixel coords
        scale: optional scaling factor to multiply predicted coords (e.g., if
               working with downsampled heatmaps). Could be a scalar or matching shape.

    Returns:
        mse, rmse, mae, euclid_dist: floats computed over all valid frames
                                     (those where target != (0,0)).
        If there are no valid frames, returns 0 for each metric.
    """

    pred_x_logits, pred_y_logits = pred_logits
    device = pred_x_logits.device

    # Move target_coords to correct device and convert to float
    target_coords = target_coords.to(device).float()  # [B, N, 2]
    B, N, W = pred_x_logits.shape
    _, _, H = pred_y_logits.shape  # same B, N, dimension = H

    # Identify valid frames: (x != 0) & (y != 0)
    # This yields a boolean mask of shape [B, N]
    valid_mask = (target_coords[..., 0] != 0) | (target_coords[..., 1] != 0)
    # NOTE: If you want (x != 0 AND y != 0) to be valid, do '&' instead of '|'

    # Find argmax along width/height => predicted indices => shape [B, N]
    x_pred_indices = torch.argmax(pred_x_logits, dim=2)
    y_pred_indices = torch.argmax(pred_y_logits, dim=2)

    x_pred = x_pred_indices.float()
    y_pred = y_pred_indices.float()

    # Stack => shape [B, N, 2]
    pred_coords = torch.stack([x_pred, y_pred], dim=2)

    # Optionally apply scaling
    if scale is not None:
        pred_coords = pred_coords * scale

    # Compute difference => shape [B, N, 2]
    diff = pred_coords - target_coords

    # Convert valid_mask from [B, N] to [B, N, 1] for broadcast
    valid_mask_expanded = valid_mask.unsqueeze(-1)  # shape [B, N, 1]

    # Filter out invalid frames by setting diff to 0 where invalid
    # Alternatively, we can gather only valid frames, but let's keep shape
    diff_valid = diff * valid_mask_expanded  # invalid frames => diff=0

    # Count valid frames
    valid_count = valid_mask.sum().item()
    if valid_count == 0:
        # No valid frames => return zeros
        return 0.0, 0.0, 0.0, 0.0

    # MSE per frame => mean over last dim => shape [B, N]
    mse_per_frame = torch.mean(diff_valid ** 2, dim=2)
    # For invalid frames => the result is 0, so sum & divide by valid_count only
    mse = mse_per_frame.sum() / valid_count

    # RMSE per frame => sqrt(MSE per frame) => shape [B, N]
    rmse_per_frame = torch.sqrt(mse_per_frame)
    rmse = rmse_per_frame.sum() / valid_count

    # MAE per frame => mean(|diff|) => shape [B, N]
    mae_per_frame = torch.mean(torch.abs(diff_valid), dim=2)
    mae = mae_per_frame.sum() / valid_count

    # Euclidean distance per frame => norm => shape [B, N]
    euclid_per_frame = torch.norm(diff_valid, dim=2)
    euclid_dist = euclid_per_frame.sum() / valid_count

    return mse.item(), rmse.item(), mae.item(), euclid_dist.item()

def precision_recall_f1(pred_heatmap, target_coords, threshold=0.5):
    """
    Calculates precision, recall, and F1 score for a tracking model with separate x and y heatmaps.
    
    Args:
    - pred_heatmap: Tuple of tensors (x_heatmap, y_heatmap) of shapes [B, W] and [B, H] respectively.
    - target_coords: Tensor of shape [B, 2] with ground truth (x, y) coordinates.
    - threshold: Float, threshold to binarize predictions.
    
    Returns:
    - precision: Float, precision across the batch.
    - recall: Float, recall across the batch.
    - f1_score: Float, F1 score across the batch.
    """
    x_heatmap, y_heatmap = pred_heatmap
    B, W = x_heatmap.shape
    _, H = y_heatmap.shape
    device = x_heatmap.device

    # Convert target coordinates to one-hot encoded x and y heatmaps
    x_target = torch.zeros_like(x_heatmap, device=device)
    y_target = torch.zeros_like(y_heatmap, device=device)
    
    # Extract x and y coordinates from target_coords
    x_coords = target_coords[:, 0].long()  # x-coordinates
    y_coords = target_coords[:, 1].long()  # y-coordinates

    # Create one-hot encoding for target positions
    x_target[torch.arange(B), x_coords] = 1
    y_target[torch.arange(B), y_coords] = 1

    # Binarize predicted heatmaps based on the threshold
    x_pred_binary = (x_heatmap >= threshold).float()
    y_pred_binary = (y_heatmap >= threshold).float()

    # Calculate true positives, false positives, and false negatives for x and y dimensions
    true_positives_x = (x_pred_binary * x_target).sum(dim=1)
    predicted_positives_x = x_pred_binary.sum(dim=1)
    actual_positives_x = x_target.sum(dim=1)

    true_positives_y = (y_pred_binary * y_target).sum(dim=1)
    predicted_positives_y = y_pred_binary.sum(dim=1)
    actual_positives_y = y_target.sum(dim=1)

    # Calculate precision, recall, and F1 score for x and y axes separately
    precision_x = true_positives_x / (predicted_positives_x + 1e-8)
    recall_x = true_positives_x / (actual_positives_x + 1e-8)
    f1_score_x = 2 * (precision_x * recall_x) / (precision_x + recall_x + 1e-8)

    precision_y = true_positives_y / (predicted_positives_y + 1e-8)
    recall_y = true_positives_y / (actual_positives_y + 1e-8)
    f1_score_y = 2 * (precision_y * recall_y) / (precision_y + recall_y + 1e-8)

    # Average metrics across the batch and axes
    precision = (precision_x + precision_y).mean().item() / 2
    recall = (recall_x + recall_y).mean().item() / 2
    f1_score = (f1_score_x + f1_score_y).mean().item() / 2

    return precision, recall, f1_score


def precision_recall_f1_tracknet_mimo(pred_coords, target_coords, distance_threshold=5.0):
    """
    Calculates precision, recall, F1 score, and accuracy for TrackNet-style tracking 
    when both predicted and target coordinates have shape [B, N, 2]:
      - B = batch size
      - N = number of frames
      - 2 = (x, y) coordinates

    A detection is considered a true positive if:
      - The ground truth (target) is not (0, 0) for that frame
      - The predicted coords are not (0, 0)
      - The Euclidean distance <= distance_threshold

    Args:
        pred_coords:   [B, N, 2] predicted (x, y) coordinates for each frame
        target_coords: [B, N, 2] ground truth (x, y) coordinates
        distance_threshold: float, maximum distance for a detection to be a true positive

    Returns:
        precision, recall, f1_score, accuracy: floats computed over the entire batch+frames
    """
    device = pred_coords.device
    # 1) Identify non-zero frames
    #    (x != 0) & (y != 0) => "ball is in frame"
    pred_nonzero = (pred_coords[..., 0] != 0) & (pred_coords[..., 1] != 0)   # [B, N]
    target_nonzero = (target_coords[..., 0] != 0) & (target_coords[..., 1] != 0) # [B, N]

    # 2) Compute Euclidean distances for every (b, n)
    #    shape: [B, N]
    distances = torch.norm(pred_coords - target_coords, dim=2)

    # 3) Define True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN)
    # TP: both nonzero + distance <= threshold
    tp_mask = target_nonzero & pred_nonzero & (distances <= distance_threshold)
    tp = tp_mask.sum().float()

    # FP: both nonzero + distance > threshold
    fp_mask = target_nonzero & pred_nonzero & (distances > distance_threshold)
    fp = fp_mask.sum().float()

    # FN: target nonzero, prediction zero
    fn_mask = target_nonzero & (~pred_nonzero)
    fn = fn_mask.sum().float()

    # TN: target zero, prediction zero
    tn_mask = (~target_nonzero) & (~pred_nonzero)
    tn = tn_mask.sum().float()

    eps = 1e-8
    # 4) Compute Precision, Recall, F1
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = 2.0 * (precision * recall) / (precision + recall + eps)

    # 5) Compute Accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return precision.item(), recall.item(), f1_score.item(), accuracy.item()

def precision_recall_f1_tracknet(pred_coords, target_coords, distance_threshold=5):
    """
    Calculates precision, recall, and F1 score for TrackNet-style tracking,
    where a detection is counted as true positive if within a certain distance
    threshold from the ground truth.
    
    Args:
    - pred_coords: Tensor of shape [B, 2] with predicted (x, y) coordinates.
    - target_coords: Tensor of shape [B, 2] with ground truth (x, y) coordinates.
    - distance_threshold: Float, maximum allowed distance for a detection to be considered a true positive.
    
    Returns:
    - precision: Float, precision across the batch.
    - recall: Float, recall across the batch.
    - f1_score: Float, F1 score across the batch.
    """

    # Calculate Euclidean distances between predicted and target coordinates
    distances = torch.norm(pred_coords - target_coords, dim=1)  # Shape: [B]

    # Determine true positives, false positives, false negatives, and true negatives
    tp = ((distances <= distance_threshold) & (target_coords != (0, 0))).sum().float()  # True positives: ball is within the threshold and ball is in the frame
    fp = ((distances > distance_threshold) & (target_coords != (0, 0))).sum().float()   # False positives: ball is in the frame but prediction is not correct
    fn = ((target_coords != (0, 0)) & (pred_coords == (0, 0)))          # False negatives: ball is in the frame, but prediction says it’s not
    tn = ((target_coords == (0, 0)) & (pred_coords == (0, 0)))          # True negatives: ball is not in the frame, and prediction correctly says so

    # Precision, recall, and F1 score calculations
    precision = tp / (tp + fp + 1e-8)  # Adding epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Accuracy calculation
    accuracy = tp / (tp + fp + fn + 1e-8)

    return precision.item(), recall.item(), f1_score.item(), accuracy.item()

def heatmap_calculate_metrics_2d(pred_heatmap, target_coords, scale=None):
    """
    Calculates evaluation metrics between predicted heatmap and target pixel coordinates.

    Args:
    - pred_heatmap: Tensor of shape [B, H, W] with predicted heatmap values
    - target_coords: Tensor of shape [B, 2] with ground truth (x, y) coordinates (integers)
    - scale: Tensor (optional), scaling factor for predicted coordinates

    Returns:
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
    - euclidean_distance: Average Euclidean distance between predictions and ground truth
    """
    # Ensure tensors are on the same device
    device = pred_heatmap.device
    target_coords = target_coords.to(device)

    # Get height and width from the heatmap shape
    B, H, W = pred_heatmap.shape

    # Flatten the heatmap to find the index of the maximum value
    pred_flat = pred_heatmap.view(B, -1)  # Shape: [B, H*W]
    max_indices = torch.argmax(pred_flat, dim=1)  # Shape: [B]

    # Convert flat indices to 2D coordinates (y, x)
    y_pred = (max_indices // W).float()  # Divide by width to get y-coordinate
    x_pred = (max_indices % W).float()   # Modulus by width to get x-coordinate

    # Stack the predicted x and y coordinates
    pred_coords = torch.stack([x_pred, y_pred], dim=1)  # Shape: [B, 2]

    # Apply scaling if scale tensor is provided
    if scale is not None:
        pred_coords = pred_coords * scale

    # Convert target coordinates to float
    target_coords = target_coords.float()

    # Difference between predicted and ground truth coordinates
    diff = pred_coords - target_coords  # Shape: [B, 2]

    # Mean Squared Error (MSE) per sample
    mse_per_sample = torch.mean(diff ** 2, dim=1)  # Shape: [B]
    mse = mse_per_sample.mean()  # Mean MSE over all samples

    # Root Mean Squared Error (RMSE) per sample
    rmse_per_sample = torch.sqrt(mse_per_sample)  # Shape: [B]
    rmse = rmse_per_sample.mean()  # Mean RMSE over all samples

    # Mean Absolute Error (MAE) per sample
    mae_per_sample = torch.mean(torch.abs(diff), dim=1)  # Shape: [B]
    mae = mae_per_sample.mean()  # Mean MAE over all samples

    # Euclidean distance per sample
    euclidean_distance_per_sample = torch.norm(diff, dim=1)  # Shape: [B]
    euclidean_distance = euclidean_distance_per_sample.mean()  # Mean Euclidean distance over all samples

    return mse.item(), rmse.item(), mae.item(), euclidean_distance.item()

# Example code to calculate RMSE for a single sample
def calculate_rmse(original_x, original_y, rescaled_x_pred, rescaled_y_pred):
    # Calculate the squared differences
    x_diff = (rescaled_x_pred - original_x) ** 2
    y_diff = (rescaled_y_pred - original_y) ** 2

    # Sum the squared differences
    squared_error = x_diff + y_diff

    # Take the square root to compute RMSE
    rmse = torch.sqrt(squared_error)

    return rmse.item()

def calculate_rmse_batched(pred_coords, label_coords):
    """
    Calculates the RMSE between predicted coordinates and ground truth labels for a batch.

    Args:
        pred_coords (tensor): Predicted coordinates of shape [B, 2].
        label_coords (tensor): Ground truth coordinates of shape [B, 2].

    Returns:
        rmse (float): Mean RMSE across the batch.
    """

    # Ensure both tensors are on the same device
    pred_coords = pred_coords.to(label_coords.device)

    # Calculate the squared differences for x and y
    squared_diff = (pred_coords - label_coords) ** 2  # [B, 2]

    # Sum the squared differences along the coordinate axis (x and y)
    sum_squared_diff = torch.sum(squared_diff, dim=-1)  # [B]

    # Take the square root to get RMSE for each sample
    rmse_per_sample = torch.sqrt(sum_squared_diff)  # [B]

    # Calculate the mean RMSE across the batch
    mean_rmse = torch.mean(rmse_per_sample)

    return mean_rmse.item()


def classification_metrics(preds, labels, num_classes=2):
    """
    Calculate accuracy, precision, recall, and F1-score for classification.

    Args:
        preds (torch.Tensor): Predictions of shape [B, 2] (logits or probabilities).
        labels (torch.Tensor): Ground truth of shape [B, 2] (one-hot encoded).
        num_classes (int): Number of classes.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    # Convert one-hot encoded labels and predictions to class indices
    predicted_classes = torch.argmax(preds, dim=1)  # Shape [B]
    true_classes = torch.argmax(labels, dim=1)  # Shape [B]

    # Calculate accuracy
    correct = (predicted_classes == true_classes).sum().item()
    total = true_classes.size(0)
    accuracy = correct / total

    # Confusion matrix
    confusion_matrix = torch.zeros((num_classes, num_classes), device=preds.device)
    for t, p in zip(true_classes, predicted_classes):
        confusion_matrix[t.long(), p.long()] += 1

    # True positives, false positives, false negatives
    true_positives = torch.diag(confusion_matrix)
    false_positives = confusion_matrix.sum(dim=0) - true_positives
    false_negatives = confusion_matrix.sum(dim=1) - true_positives

    # Precision, recall, F1-score
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Macro averages
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1
    }



def classification_metrics_class_1(preds, labels):
    """
    Calculate precision, recall, accuracy, and F1-score for class 1.

    Args:
        preds (torch.Tensor): Predictions of shape [B, 2] (logits or probabilities).
        labels (torch.Tensor): Ground truth of shape [B, 2] (one-hot encoded).

    Returns:
        dict: A dictionary containing precision, recall, accuracy, and F1-score for class 1.
    """
    # Convert one-hot encoded labels and predictions to class indices
    predicted_classes = torch.argmax(preds, dim=1)  # Shape [B]
    true_classes = torch.argmax(labels, dim=1)  # Shape [B]

    # Confusion matrix for class 1
    true_positives = ((predicted_classes == 1) & (true_classes == 1)).sum().item()
    false_positives = ((predicted_classes == 1) & (true_classes == 0)).sum().item()
    false_negatives = ((predicted_classes == 0) & (true_classes == 1)).sum().item()
    true_negatives = ((predicted_classes == 0) & (true_classes == 0)).sum().item()

    # Precision, recall, F1-score, and accuracy for class 1
    precision_1 = true_positives / (true_positives + false_positives + 1e-8)
    recall_1 = true_positives / (true_positives + false_negatives + 1e-8)
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + 1e-8)
    accuracy_1 = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    return {
        "accuracy": accuracy_1,
        "precision": precision_1,
        "recall": recall_1,
        "f1_score": f1_1,
    }


def post_process_event_prediction(preds):
    _, predicted_classes = torch.max(preds, dim=1)  # [B]

    return predicted_classes



def PCE(sample_prediction_events, sample_target_events):
    """
    Percentage of Correct Events for PyTorch tensors.

    :param sample_prediction_events: Predicted events, tensor of shape [2,]
    :param sample_target_events: Ground truth events, tensor of shape [2,]
    :return: Integer (1 for correct, 0 for incorrect)
    """
    # Threshold predictions and targets
    sample_prediction_events = (sample_prediction_events >= 0.5).float()
    sample_target_events = (sample_target_events >= 0.5).float()
    
    # Compute the difference
    diff = sample_prediction_events - sample_target_events
    
    # Check if all values are correct
    if torch.sum(diff) != 0:  # Incorrect
        ret_pce = 0
    else:  # Correct
        ret_pce = 1
    return ret_pce


def SPCE(sample_prediction_events, sample_target_events, thresh=0.25):
    """
    Smooth Percentage of Correct Events for PyTorch tensors.

    :param sample_prediction_events: Predicted events, tensor of shape [2,]
    :param sample_target_events: Ground truth events, tensor of shape [2,]
    :param thresh: Threshold for the difference between prediction and ground truth
    :return: Integer (1 for correct, 0 for incorrect)
    """
    # Compute the absolute difference
    diff = torch.abs(sample_prediction_events - sample_target_events)
    
    # Check if all differences are within the threshold
    if torch.sum(diff > thresh) > 0:  # Incorrect
        ret_spce = 0
    else:  # Correct
        ret_spce = 1
    return ret_spce


def batch_PCE(batch_prediction_events, batch_target_events):
    """
    Batch Percentage of Correct Events (PCE) using PyTorch tensors.
    
    :param batch_prediction_events: Batch of predictions, size: (B, N)
    :param batch_target_events: Batch of ground truths, size: (B, N)
    :return: Tensor of PCE values for each sample in the batch, size: (B,)
    """
    # Threshold predictions and targets
    batch_prediction_events = (batch_prediction_events >= 0.5).float()
    batch_target_events = (batch_target_events >= 0.5).float()
    
    # Compute difference
    diff = batch_prediction_events - batch_target_events  # Shape: (B, N)
    
    # Check correctness for each sample
    batch_pce = (diff.abs().sum(dim=1) == 0).float()  # 1 if correct, 0 if incorrect
    
    return batch_pce.mean()


def batch_SPCE(batch_prediction_events, batch_target_events, thresh=0.25):
    """
    Batch Smooth Percentage of Correct Events (SPCE) using PyTorch tensors.
    
    :param batch_prediction_events: Batch of predictions, size: (B, N)
    :param batch_target_events: Batch of ground truths, size: (B, N)
    :param thresh: Threshold for the difference between prediction and ground truth.
    :return: Tensor of SPCE values for each sample in the batch, size: (B,)
    """
    # Compute absolute difference
    diff = torch.abs(batch_prediction_events - batch_target_events)  # Shape: (B, N)
    
    # Check if all differences are within the threshold for each sample
    batch_spce = (diff <= thresh).all(dim=1).float()  # 1 if all within threshold, 0 otherwise
    
    return batch_spce.mean()



def pck_calculation(
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor,
        thresholds,
        norm: Optional[Union[float, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ):
    """
    Calculate Percentage of Correct Keypoints (PCK) over one or many thresholds.

    Args:
        pred_coords (torch.Tensor): Predicted coords, shape [B, 2].
        target_coords (torch.Tensor): Ground-truth coords, shape [B, 2].
        thresholds (list | tuple | torch.Tensor | float): Distance threshold(s).
            If multiple, returns a PCK value per threshold.
        norm (float | torch.Tensor | None): Optional normalization factor.
            - If float: distances are divided by this scalar.
            - If tensor of shape [B]: per-sample normalization.
            - If None: raw pixel distances are used.
        mask (torch.Tensor | None): Optional boolean mask of shape [B].
            True keeps a sample; False excludes it from PCK.

    Returns:
        dict with:
            - 'pck': torch.Tensor of shape [T], PCK at each threshold.
            - 'thresholds': torch.Tensor of shape [T], thresholds used.
            - 'distances': torch.Tensor of shape [N], distances actually evaluated (after masking & normalization).
            - 'num_samples': int, number of samples used.
    """
    # ensure tensors & dtypes
    pred = pred_coords.float()
    tgt  = target_coords.float()

    # pairwise Euclidean distances (B,)
    dists = torch.linalg.norm(pred - tgt, dim=-1)  # sqrt(dx^2 + dy^2)

    # optional normalization
    if norm is not None:
        if not torch.is_tensor(norm):
            norm = torch.tensor(norm, dtype=dists.dtype, device=dists.device)
        dists = dists / norm

    # optional mask
    if mask is not None:
        keep = mask.bool()
        dists = dists[keep]

    # guard: no samples left
    if dists.numel() == 0:
        thr = torch.as_tensor(thresholds, dtype=torch.float32)
        return {
            "pck": torch.zeros_like(thr),
            "thresholds": thr,
            "distances": dists,
            "num_samples": 0
        }

    # thresholds -> tensor on same device
    thr = torch.as_tensor(thresholds, dtype=dists.dtype, device=dists.device).flatten()

    # vectorized correctness: (N,1) <= (1,T) -> (N,T)
    correct = (dists[:, None] <= thr[None, :]).float()

    # PCK per threshold (mean over samples)
    pck = correct.mean(dim=0)

    result = {
        "pck": pck,                 # [T]
        "thresholds": thr,          # [T]
        "distances": dists,         # [N]
        "num_samples": int(dists.numel())
    }

    print(result)


    return result


def print_pck_results(pck_out, title="PCK Results"):
    """
    Nicely print PCK results from pck_calculation.
    """
    pck = pck_out["pck"].cpu().numpy()
    thresholds = pck_out["thresholds"].cpu().numpy()
    n = pck_out["num_samples"]

    print(f"\n=== {title} ===")
    print(f"Evaluated on {n} samples")
    print("-" * 30)
    print(f"{'Threshold':>10} | {'PCK':>6}")
    print("-" * 30)
    for t, v in zip(thresholds, pck):
        print(f"{t:10.2f} | {v*100:5.2f}%")
    print("-" * 30)
    print(f"AUC-PCK (1..{int(thresholds.max())}): "
          f"{np.trapz(pck, thresholds)/(thresholds.max()-thresholds.min()):.3f}")