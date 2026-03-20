import torch.nn as nn
import torch
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

class Heatmap_Ball_Detection_Loss_Weighted_MultiFrame(nn.Module):
    def __init__(self, weighted_list=[1, 2, 2, 3], sigma=0.3):
        """
        Args:
            weighted_list: List of 4 weights, one for each visibility class [0..3].
            sigma: Std dev for Gaussian smoothing (if visibility == 3).
        """
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction="none")  # We'll handle reduction manually
        self.weighted_list = weighted_list
        self.sigma = sigma

    def forward(self, output, target_ball_position, visibility):
        """
        Args:
            output: tuple (pred_x, pred_y) each of shape [B, N, W] or [B, N, H]
                    B = batch size, N = number of frames, W/H = width/height
            target_ball_position: [B, N, 2] integer (x, y) pixel coords
            visibility: [B, N] integer visibility per frame [0..3]
        Returns:
            A scalar loss (float)
        """
        pred_x, pred_y = output  # pred_x: [B, N, W], pred_y: [B, N, H]

        device = pred_x.device
        B, N, W = pred_x.shape
        _, _, H = pred_y.shape  # Same B, N, but dimension is H

        # Convert target coords to int and clamp
        target_x = target_ball_position[..., 0].long().clamp_(0, W - 1)  # [B, N]
        target_y = target_ball_position[..., 1].long().clamp_(0, H - 1)  # [B, N]

        # Build zero-maps for x and y
        target_x_map = torch.zeros_like(pred_x)  # [B, N, W]
        target_y_map = torch.zeros_like(pred_y)  # [B, N, H]

        # Identify frames that are valid (not (0,0)) => skip_mask is True if we skip
        skip_mask = (target_ball_position[..., 0] == 0) & (target_ball_position[..., 1] == 0)

        # Identify frames that want Gaussian smoothing (visibility == 3)
        gaussian_mask = (visibility == 3) & (~skip_mask)  # Must be valid + vis==3

        # 1) Create one-hot target for non-gaussian frames
        # We'll do a for-loop for clarity, but you can vectorize if desired
        for b in range(B):
            for n in range(N):
                if skip_mask[b, n]:
                    continue  # Skip frames with (0,0)
                if not gaussian_mask[b, n]:
                    # Place a 1 at [b, n, target_x[b,n]] or [b, n, target_y[b,n]]
                    tx = target_x[b, n]
                    ty = target_y[b, n]
                    target_x_map[b, n, tx] = 1.0
                    target_y_map[b, n, ty] = 1.0

        # 2) Create Gaussian targets for frames with vis=3
        #    Then normalize each row to sum to 1
        if gaussian_mask.any():
            # Prepare coordinate grids
            x_coords = torch.arange(W, device=device).float()  # [W]
            y_coords = torch.arange(H, device=device).float()  # [H]

            for b in range(B):
                for n in range(N):
                    if not gaussian_mask[b, n]:
                        continue
                    tx = target_x[b, n].float()
                    ty = target_y[b, n].float()

                    # Build Gaussian along X
                    dx = x_coords - tx  # [W]
                    gauss_x = torch.exp(-0.5 * (dx**2) / (self.sigma**2))  # shape [W]

                    # Build Gaussian along Y
                    dy = y_coords - ty  # [H]
                    gauss_y = torch.exp(-0.5 * (dy**2) / (self.sigma**2))  # shape [H]

                    # Normalize each so sum=1
                    gauss_x = gauss_x / (gauss_x.sum() + 1e-8)
                    gauss_y = gauss_y / (gauss_y.sum() + 1e-8)

                    # Place them in target_x_map / target_y_map
                    target_x_map[b, n] = gauss_x
                    target_y_map[b, n] = gauss_y

        # 3) Compute BCELoss w/o reduction
        loss_x = self.loss_fn(pred_x, target_x_map)  # shape [B, N, W]
        loss_y = self.loss_fn(pred_y, target_y_map)  # shape [B, N, H]

        # Sum along W/H => [B, N]
        loss_x = loss_x.sum(dim=2)
        loss_y = loss_y.sum(dim=2)

        # Combine or sum => shape [B, N]
        loss_per_frame = loss_x + loss_y  # [B, N]

        # Convert visibility to weights
        weights_tensor = torch.tensor(self.weighted_list, device=device, dtype=torch.float)
        frame_weights = weights_tensor[visibility]  # shape [B, N]

        # Zero out skip frames entirely so they do not contribute
        # Alternatively, you can just do skip_mask => loss_per_frame=0
        loss_per_frame[skip_mask] = 0.0
        frame_weights[skip_mask] = 0.0  # or 0.0 to exclude from sum

        # Weighted loss
        loss_weighted = loss_per_frame * frame_weights  # [B, N]

        # Average across all valid frames (frames that are not skip_mask)
        valid_mask = ~skip_mask
        total_valid = valid_mask.sum().item()
        if total_valid == 0:
            # Edge case: no valid frames => return 0
            return torch.tensor(0.0, device=device, requires_grad=True)

        final_loss = loss_weighted.sum() / total_valid

        return final_loss


class HeatmapBallDetectionLoss2DWeighted(nn.Module):
    def __init__(
        self,
        H: int,
        W: int,
        weighted_list = [1.0, 2.0, 2.0, 3.0],
        sigmas = [0.8, 0.8, 1.0, 2.5],  # v0,v1,v2: tight; v3 (occluded): wide
        use_logits: bool = False,
        eps: float = 1e-7,
    ):
        """
        Args:
            H, W: spatial size of the heatmap.
            weighted_list: per-visibility weights [w0,w1,w2,w3].
            sigmas: per-visibility Gaussian std (pixels) [σ0,σ1,σ2,σ3].
            use_logits: if True, expects logits + uses BCEWithLogitsLoss; else probabilities + BCELoss.
            eps: clamp for BCELoss stability.
        """
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.weighted_list = weighted_list
        self.sigmas = sigmas
        self.use_logits = use_logits
        self.eps = eps
        self.loss = nn.BCEWithLogitsLoss(reduction="none") if use_logits else nn.BCELoss(reduction="none")

        # Precompute coordinate grids as buffers (broadcast-friendly shapes)
        # Shapes: xs[1,1,W], ys[1,H,1]
        self.register_buffer("xs", torch.arange(self.W).view(1, 1, self.W).float())
        self.register_buffer("ys", torch.arange(self.H).view(1, self.H, 1).float())

    def forward(self, pred_map: torch.Tensor, target_ball_position: torch.Tensor, visibility: torch.Tensor):
        """
        Args:
            pred_map: [B, H*W] flattened heatmap (probabilities or logits).
            target_ball_position: [B, 2] with (x, y) in pixel coords (can be float).
            visibility: [B] in {0,1,2,3}.
        Returns:
            scalar loss
        """
        if pred_map.dim() != 2 or pred_map.size(1) != self.H * self.W:
            raise ValueError(f"pred_map must be [B, {self.H*self.W}], got {list(pred_map.shape)}")

        B = pred_map.size(0)
        device = pred_map.device
        dtype  = pred_map.dtype

        # Build per-sample Gaussian targets with per-visibility sigma
        # target_map -> [B, H, W]
        target_map = torch.zeros((B, self.H, self.W), device=device, dtype=dtype)

        # Coords
        tx = target_ball_position[:, 0].clamp_(0, self.W - 1).to(dtype)  # [B]
        ty = target_ball_position[:, 1].clamp_(0, self.H - 1).to(dtype)  # [B]

        # Per-sample sigma from visibility
        vis_w = torch.tensor(self.weighted_list, device=device, dtype=dtype)  # [4]
        sigma_lut = torch.tensor(self.sigmas, device=device, dtype=dtype)     # [4]
        sigma_vec = sigma_lut[visibility]                                     # [B]
        denom = (2.0 * (sigma_vec ** 2)).view(B, 1, 1)                        # [B,1,1]

        # Broadcast-friendly centers
        gx = tx.view(B, 1, 1)  # [B,1,1]
        gy = ty.view(B, 1, 1)  # [B,1,1]

        # Ensure grids match dtype/device
        xs = self.xs.to(device=device, dtype=dtype).expand(B, 1, self.W)   # [B,1,W]
        ys = self.ys.to(device=device, dtype=dtype).expand(B, self.H, 1)   # [B,H,1]

        # Gaussian map for every sample (vectorized)
        dx2 = (xs - gx) ** 2          # [B,1,W]
        dy2 = (ys - gy) ** 2          # [B,H,1]
        gmap = torch.exp(-(dx2 + dy2) / denom)   # broadcast -> [B,H,W]

        # Normalize each sample to sum=1
        gsum = gmap.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        target_map = gmap / gsum

        # Flatten target to [B, H*W]
        target_flat = target_map.view(B, self.H * self.W)

        # Clamp preds if using probabilities
        if not self.use_logits:
            pred_map = pred_map.clamp(self.eps, 1.0 - self.eps)

        # Per-element BCE -> per-sample -> weighted batch mean
        per_elem = self.loss(pred_map, target_flat)     # [B, H*W]
        per_sample = per_elem.sum(dim=1)                # [B]

        sample_weights = vis_w[visibility]              # [B]
        loss = (per_sample * sample_weights).mean()
        return loss

        

class Heatmap_Ball_Detection_Loss_Weighted(nn.Module):
    def __init__(self, weighted_list=[1, 2, 2, 3], sigma=0.3):
        """
        Args:
        - weighted_list: List of weights corresponding to the four visibility classes.
        """
        super(Heatmap_Ball_Detection_Loss_Weighted, self).__init__()
        self.loss = nn.BCELoss(reduction="none")  # Avoid reduction for per-sample weighting
        self.weighted_list = weighted_list
        self.sigma = sigma

    def forward(self, output, target_ball_position, visibility):
        """
        Args:
        - output: tuple of (pred_x, pred_y)
            - pred_x: [B, W] predicted logits across the width (x-axis)
            - pred_y: [B, H] predicted logits across the height (y-axis)
        - target_ball_position: [B, 2] true (x, y) integer pixel coordinates of the ball
        - visibility: [B] visibility labels for the ball (0 to 3 corresponding to visibility classes).
        """
        # Correctly unpack the output logits
        pred_x, pred_y = output

        # Ensure target positions are of type LongTensor and on the same device
        device = pred_x.device
        target_x = target_ball_position[:, 0].long().to(device)  # [B]
        target_y = target_ball_position[:, 1].long().to(device)  # [B]

        # Clamp the indices to valid ranges
        target_x = torch.clamp(target_x, 0, pred_x.shape[1] - 1)
        target_y = torch.clamp(target_y, 0, pred_y.shape[1] - 1)

        gaussian_mask = (visibility == 3)

        # Gaussian-based target generation
        if gaussian_mask.any():
            x_coords = torch.arange(pred_x.shape[1], device=device).float()  # Width
            y_coords = torch.arange(pred_y.shape[1], device=device).float()  # Height

            target_x_map_gaussian = torch.exp(
                -((x_coords.unsqueeze(0).unsqueeze(0) - target_x[gaussian_mask].unsqueeze(1).unsqueeze(2)) ** 2)
                / (2 * self.sigma ** 2)
            )  # Shape: [masked_B, 1, W]

            target_y_map_gaussian = torch.exp(
                -((y_coords.unsqueeze(0).unsqueeze(0) - target_y[gaussian_mask].unsqueeze(1).unsqueeze(2)) ** 2)
                / (2 * self.sigma ** 2)
            )  # Shape: [masked_B, 1, H]

       
        # For x-axis predictions
        target_x_map = torch.zeros_like(pred_x)  # [B, W]
        target_x_map.scatter_(1, target_x.unsqueeze(1), 1.0)

        # For y-axis predictions
        target_y_map = torch.zeros_like(pred_y)  # [B, H]
        target_y_map.scatter_(1, target_y.unsqueeze(1), 1.0)

        # Replace targets with Gaussian maps where applicable
        if gaussian_mask.any():
            target_x_map[gaussian_mask] = target_x_map_gaussian.squeeze(1)  # Replace for Gaussian targets
            target_y_map[gaussian_mask] = target_y_map_gaussian.squeeze(1)

            # Normalize each map to sum to 1
            target_x_map[gaussian_mask] /= target_x_map[gaussian_mask].sum(dim=-1, keepdim=True)
            target_y_map[gaussian_mask] /= target_y_map[gaussian_mask].sum(dim=-1, keepdim=True)

        # Compute binary cross-entropy loss for x and y without reduction
        loss_x = self.loss(pred_x, target_x_map).sum(dim=1)  # Sum across width [B]
        loss_y = self.loss(pred_y, target_y_map).sum(dim=1)  # Sum across height [B]

        # Compute per-sample weights based on visibility classes
        visibility_weights = torch.tensor(self.weighted_list, device=device)  # [4]
        sample_weights = visibility_weights[visibility]  # Map visibility to weights [B]

        # Apply weights to the losses
        weighted_loss_x = (loss_x * sample_weights).mean()  # Average across batch
        weighted_loss_y = (loss_y * sample_weights).mean()  # Average across batch

        # Return the combined weighted loss
        return weighted_loss_x + weighted_loss_y



class Heatmap_Ball_Detection_Loss(nn.Module):
    def __init__(self):
        super(Heatmap_Ball_Detection_Loss, self).__init__()
        self.loss = nn.BCELoss() # Use BCEWithLogitsLoss for logits

    def forward(self, output, target_ball_position, visibility):
        """
        Args:
        - output: tuple of (pred_x, pred_y)
            - pred_x: [B, W] predicted logits across the width (x-axis)
            - pred_y: [B, H] predicted logits across the height (y-axis)
        - target_ball_position: [B, 2] true (x, y) integer pixel coordinates of the ball
        """
        # Correctly unpack the output logits
        pred_x, pred_y = output

        # Ensure target positions are of type LongTensor and on the same device
        device = pred_x.device
        target_x = target_ball_position[:, 0].long().to(device)  # [B]
        target_y = target_ball_position[:, 1].long().to(device)  # [B]

        # Clamp the indices to valid ranges
        target_x = torch.clamp(target_x, 0, pred_x.shape[1] - 1)
        target_y = torch.clamp(target_y, 0, pred_y.shape[1] - 1)

        # For x-axis predictions
        target_x_one_hot = torch.zeros_like(pred_x)  # [B, W]
        target_x_one_hot.scatter_(1, target_x.unsqueeze(1), 1.0)

        # For y-axis predictions
        target_y_one_hot = torch.zeros_like(pred_y)  # [B, H]
        target_y_one_hot.scatter_(1, target_y.unsqueeze(1), 1.0)

        # Compute binary cross-entropy loss for x and y
        loss_x = self.loss(pred_x, target_x_one_hot)
        loss_y = self.loss(pred_y, target_y_one_hot)

        # Return the combined loss
        return loss_x + loss_y

class Heatmap_Ball_Detection_Loss_Gaussian(nn.Module):
    def __init__(self, sigma=0.25, weighted_list=[1,2,3,3]):

        super(Heatmap_Ball_Detection_Loss_Gaussian, self).__init__()
        self.loss = nn.BCELoss(reduction='none') # More stable for logits
        self.sigma = sigma  # Standard deviation for Gaussian spread
        self.weighted_list = weighted_list

    def forward(self, output, target_ball_position, visibility):
        """
        Args:
        - output: tuple of (pred_x, pred_y)
            - pred_x: [B, W] predicted logits across the width (x-axis)
            - pred_y: [B, H] predicted logits across the height (y-axis)
        - target_ball_position: [B, 2] true (x, y) integer pixel coordinates of the ball
        """
        pred_x, pred_y = output
        device = pred_x.device
        target_x = target_ball_position[:, 0].long().to(device)
        target_y = target_ball_position[:, 1].long().to(device)

        # Clamp to valid ranges
        target_x = torch.clamp(target_x, 0, pred_x.shape[1] - 1)
        target_y = torch.clamp(target_y, 0, pred_y.shape[1] - 1)

        # Create coordinate ranges for Gaussian distribution
        x_coords = torch.arange(pred_x.shape[1], device=device).float()  # Width
        y_coords = torch.arange(pred_y.shape[1], device=device).float()  # Height

        # Generate Gaussian distributions centered at the ground truth positions
        target_x_gaussian = torch.exp(-((x_coords.unsqueeze(0) - target_x.unsqueeze(1))**2) / (2 * self.sigma**2))
        target_y_gaussian = torch.exp(-((y_coords.unsqueeze(0) - target_y.unsqueeze(1))**2) / (2 * self.sigma**2))

        # Normalize to sum to 1
        target_x_gaussian /= target_x_gaussian.sum(dim=1, keepdim=True)
        target_y_gaussian /= target_y_gaussian.sum(dim=1, keepdim=True)

        #Compute per-pixel BCE loss
        loss_x = self.loss(pred_x, target_x_gaussian)  # [B, W]
        loss_y = self.loss(pred_y, target_y_gaussian)  # [B, H]

        # Sum losses across the width and height
        loss_x = loss_x.sum(dim=1)  # [B]
        loss_y = loss_y.sum(dim=1)  # [B]

        # Create mask based on visibility and weights
        visibility_weights = torch.tensor(self.weighted_list, device=device)  # [num_visibility_states]
        mask = visibility_weights[visibility]  # [B]

        # Apply mask to the losses
        weighted_loss_x = (loss_x * mask).mean()
        weighted_loss_y = (loss_y * mask).mean()

    
        return weighted_loss_x + weighted_loss_y

class Heatmap_Ball_Detection_Loss_2D(nn.Module):
    def __init__(self, h, w, sigma=2.0):
        super(Heatmap_Ball_Detection_Loss_2D, self).__init__()
        self.h = h  # Image height
        self.w = w  # Image width
        self.sigma = sigma
        self.bce_loss = nn.BCELoss()  # Use BCEWithLogitsLoss if your output is logits

    def forward(self, output, target_ball_position):
        """
        Args:
        - output: [B, H, W] predicted heatmap with probabilities for each pixel
        - target_ball_position: [B, 2] true (x, y) integer pixel coordinates of the ball
        """
        device = output.device
        batch_size = output.size(0)

        # Ensure sigma is a tensor
        sigma_tensor = torch.tensor(self.sigma, dtype=torch.float32, device=device)

        # Create coordinate grids
        y_coords = torch.arange(self.h, device=device).view(1, self.h, 1).expand(batch_size, self.h, self.w)
        x_coords = torch.arange(self.w, device=device).view(1, 1, self.w).expand(batch_size, self.h, self.w)

        # Extract target positions and reshape for broadcasting
        x_targets = target_ball_position[:, 0].view(batch_size, 1, 1).float()
        y_targets = target_ball_position[:, 1].view(batch_size, 1, 1).float()

        # Compute squared distances
        squared_distances = ((x_coords - x_targets) ** 2 + (y_coords - y_targets) ** 2)

        # Compute Gaussian heatmaps
        target_heatmap = torch.exp(-squared_distances / (2 * sigma_tensor ** 2))

        # Compute binary cross-entropy loss between the predicted and target heatmaps
        loss = self.bce_loss(output, target_heatmap)

        return loss


class HeatmapBallDetectionLoss(nn.Module):
    def __init__(self, h, w):
        super(HeatmapBallDetectionLoss, self).__init__()
        self.h = h  # Image height
        self.w = w  # Image width
        self.bce_loss = nn.BCELoss()  # Use BCEWithLogitsLoss for logits

    def forward(self, output, target_ball_position):
        """
        Args:
        - output: tuple of (pred_x, pred_y)
            - pred_x: [B, N, W] predicted logits across the width (x-axis)
            - pred_y: [B, N, H] predicted logits across the height (y-axis)
        - target_ball_position: [B, N, 2] true (x, y) integer pixel coordinates of the ball.
            [-1, -1] entries indicate missing ball positions.
        """
        # Unpack the output logits
        pred_x, pred_y = output  # [B, N, W] and [B, N, H]

        # Ensure target positions are of type LongTensor and on the same device
        device = pred_x.device
        target_x = target_ball_position[..., 0].long().to(device)  # [B, N]
        target_y = target_ball_position[..., 1].long().to(device)  # [B, N]

        # Clamp the indices to valid ranges
        target_x = torch.clamp(target_x, 0, pred_x.shape[2] - 1)  # W dimension
        target_y = torch.clamp(target_y, 0, pred_y.shape[2] - 1)  # H dimension

        # Initialize the total loss
        total_loss_x, total_loss_y = 0.0, 0.0
        valid_count = 0

        # Iterate over each frame and batch to compute the loss only on valid frames
        batch_size, num_frames = target_x.shape
        for b in range(batch_size):
            for n in range(num_frames):
                if (target_x[b, n] == 0) and (target_y[b, n] == 0):
                    # Skip invalid frames with [-1, -1] labels
                    continue

                # Create one-hot encoded ground truth for the current frame
                target_x_one_hot = torch.zeros_like(pred_x[b, n])  # [W]
                target_y_one_hot = torch.zeros_like(pred_y[b, n])  # [H]

                # Set the ground truth positions in the one-hot vectors
                target_x_one_hot[target_x[b, n]] = 1.0
                target_y_one_hot[target_y[b, n]] = 1.0

                # Compute the BCE loss for the current frame
                loss_x = self.bce_loss(pred_x[b, n], target_x_one_hot)
                loss_y = self.bce_loss(pred_y[b, n], target_y_one_hot)
        
                # Accumulate the total loss
                total_loss_x += loss_x
                total_loss_y += loss_y
                valid_count += 1

        # Avoid division by zero
        if valid_count > 0:
            total_loss_x /= valid_count
            total_loss_y /= valid_count

        # Return the combined loss
        return total_loss_x + total_loss_y


def events_spotting_loss(pred_events, target_events, weights=(1, 3), epsilon=1e-9):
    """
    Weighted binary cross-entropy loss for event spotting.

    Args:
        pred_events (torch.Tensor): Predicted probabilities, shape [B, num_events].
        target_events (torch.Tensor): Ground truth labels, shape [B, num_events].
        weights (tuple): Weights for the event classes, e.g., (1, 3).
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Loss value (scalar).
    """
    # Convert weights to a tensor and normalize
    weights = torch.tensor(weights, dtype=torch.float32).view(1, -1)
    weights = weights / weights.sum()
    
    # Move weights to the same device as predictions
    weights = weights.to(pred_events.device)
    
    # Compute the weighted binary cross-entropy loss
    loss = -torch.mean(
        weights * (
            target_events * torch.log(pred_events + epsilon) +
            (1.0 - target_events) * torch.log(1.0 - pred_events + epsilon)
        )
    )
    
    return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, epsilon=1e-9):
        """
        Binary Focal Loss to address class imbalance.

        Args:
            alpha (float): Balancing factor (higher values focus more on class 1).
            gamma (float): Focusing parameter (higher values focus more on hard examples).
            epsilon (float): Small value to avoid log(0).
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Model predictions (probabilities), shape [B,1].
            targets (torch.Tensor): Ground truth labels, shape [B,1].

        Returns:
            torch.Tensor: Computed focal loss.
        """
        preds = torch.clamp(preds, min=self.epsilon, max=1.0 - self.epsilon)  # Prevent log(0) errors

        # Compute focal loss components
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # Alpha balancing
        pt = targets * preds + (1 - targets) * (1 - preds)  # p_t for the correct class
        focal_loss = -focal_weight * ((1.0 - pt) ** self.gamma) * torch.log(pt)  # Apply focal weighting

        return focal_loss.mean()


def focal_loss(pred_events, target_events, alpha=0.6, gamma=2.0, epsilon=1e-9):
    """
    Focal loss for imbalanced datasets.

    Args:
        pred_events (torch.Tensor): Predicted probabilities, shape [B, num_events].
        target_events (torch.Tensor): Ground truth labels, shape [B, num_events].
        alpha (float): Balancing factor for positive/negative classes.
        gamma (float): Focusing parameter for hard examples.
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Loss value (scalar).
    """

    # Clamp predictions to avoid log(0) or log(1)
    pred_events = torch.clamp(pred_events, min=epsilon, max=1.0 - epsilon)

    # Compute p_t (probability for the true class)
    p_t = target_events * pred_events + (1 - target_events) * (1 - pred_events)

    # Compute alpha_t (class weights)
    alpha_t = target_events * alpha + (1 - target_events) * (1 - alpha)

    # Compute focal loss
    focal_loss = -alpha_t * ((1.0 - p_t) ** gamma) * torch.log(p_t)

    # Return the mean loss
    return focal_loss.mean()

def generate_gaussian_map(width, target_x, sigma=0.5):
    """
    Generate a 1D Gaussian map.

    Args:
        width (int): The width of the map (number of points).
        target_x (int): The center position of the Gaussian.
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        torch.Tensor: A 1D Gaussian map.
    """
    x_coords = torch.arange(width).float()  # 1D coordinate space
    gaussian_map = torch.exp(-((x_coords - target_x)**2) / (2 * sigma**2))  # Gaussian formula
    normalized_gaussian_map = gaussian_map / gaussian_map.sum()  # Normalize to sum to 1
    return normalized_gaussian_map


# Example usage
def probability_loss(pred_probs, true_probs):
    """
    Compute KL Divergence Loss.
    Args:
        pred_probs: Model output probabilities (after softmax) [B, C]
        true_probs: Ground truth probabilities [B, C]
    Returns:
        KL divergence loss.
    """
    pred_probs = pred_probs + 1e-12
    return F.kl_div(pred_probs.log(), true_probs, reduction='batchmean')



def calculate_rmse_from_heatmap(output, labels, scale=None):
    """
    Extract coordinates from the predicted logits and compute RMSE with ground truth labels.

    Args:
        output (tuple): Tuple of (pred_x, pred_y), where:
                        - pred_x: [B, W] logits across width (x-axis).
                        - pred_y: [B, H] logits across height (y-axis).
        labels (tensor): Ground truth coordinates of shape [B, 2].

    Returns:
        rmse (tensor): RMSE loss between the predicted and ground truth coordinates.
    """

    pred_x, pred_y = output  # Unpack the tuple
    B, W = pred_x.shape  # Shape of x-axis logits
    _, H = pred_y.shape  # Shape of y-axis logits

    # Ensure the labels are on the correct device
    labels = labels.to(pred_x.device)

    # Apply softmax to get probability distributions along each axis
    prob_x = torch.softmax(pred_x, dim=-1)  # [B, W]
    prob_y = torch.softmax(pred_y, dim=-1)  # [B, H]

    # Create coordinate grids for x and y axes
    x_grid = torch.linspace(0, W - 1, W, device=pred_x.device)  # [W]
    y_grid = torch.linspace(0, H - 1, H, device=pred_y.device)  # [H]

    # Compute soft-argmax to get predicted coordinates
    x_pred = torch.sum(prob_x * x_grid, dim=-1)  # [B]
    y_pred = torch.sum(prob_y * y_grid, dim=-1)  # [B]

    # Stack predicted x and y coordinates into a [B, 2] tensor
    predicted_coords = torch.stack([x_pred, y_pred], dim=-1)  # Shape: [B, 2]
    if scale is not None:
        predicted_coords = predicted_coords * scale

    # Compute RMSE between predicted coordinates and ground truth labels
    rmse = torch.sqrt(torch.mean((predicted_coords - labels) ** 2))  # RMSE loss

    return rmse
    
def extract_coords_from_heatmap(output):
    """
    Extracts the (x, y) coordinates from the heatmap logits.

    Args:
        output (tuple): Tuple of (pred_x, pred_y), where:
                        - pred_x: [B, W] logits across width (x-axis).
                        - pred_y: [B, H] logits across height (y-axis).

    Returns:
        coords (tensor): Extracted coordinates of shape [B, 2] (x, y) for each sample.
    """

    pred_x, pred_y = output  # Unpack the tuple
    B, W = pred_x.shape  # Batch size and width
    _, H = pred_y.shape  # Height

    # Apply softmax to get probability distributions along each axis
    prob_x = torch.softmax(pred_x, dim=-1)  # [B, W]
    prob_y = torch.softmax(pred_y, dim=-1)  # [B, H]

    # Create coordinate grids for x and y axes
    x_grid = torch.linspace(0, W - 1, W, device=pred_x.device)  # [W]
    y_grid = torch.linspace(0, H - 1, H, device=pred_y.device)  # [H]

    # Compute soft-argmax to get predicted coordinates
    x_pred = torch.sum(prob_x * x_grid, dim=-1)  # [B]
    y_pred = torch.sum(prob_y * y_grid, dim=-1)  # [B]

    # Stack predicted x and y coordinates into a [B, 2] tensor
    coords = torch.stack([x_pred, y_pred], dim=-1)  # Shape: [B, 2]

    return coords

def gaussian_1d(pos, muy, sigma):
    """Create 1D Gaussian distribution based on ball position (muy), and std (sigma)"""
    target = torch.exp(- (((pos - muy) / sigma) ** 2) / 2)
    return target

def create_target_ball_right(ball_position_xy, sigma, w, h, thresh_mask, device):
    """Create target for the ball detection stages

    :param ball_position_xy: Position of the ball (x,y)
    :param sigma: standard deviation (a hyperparameter)
    :param w: width of the resize image
    :param h: height of the resize image
    :param thresh_mask: if values of 1D Gaussian < thresh_mask --> set to 0 to reduce computation
    :param device: cuda() or cpu()
    :return:
    """
    w, h = int(w), int(h)
    target_ball_position_x = torch.zeros(w, device=device)
    target_ball_position_y = torch.zeros(h, device=device)
    # Only do the next step if the ball is existed
    if (w > ball_position_xy[0] > 0) and (h > ball_position_xy[1] > 0):
        # For x
        x_pos = torch.arange(0, w, device=device)
        target_ball_position_x = gaussian_1d(x_pos, ball_position_xy[0], sigma=sigma)
        # For y
        y_pos = torch.arange(0, h, device=device)
        target_ball_position_y = gaussian_1d(y_pos, ball_position_xy[1], sigma=sigma)

        target_ball_position_x[target_ball_position_x < thresh_mask] = 0.
        target_ball_position_y[target_ball_position_y < thresh_mask] = 0.

    return target_ball_position_x, target_ball_position_y


if __name__ == "__main__":
    # Initialize the loss function
    # loss_func = Heatmap_Ball_Detection_Loss_Weighted(weighted_list=[1, 2, 3, 3])

    # # Heatmap dimensions
    # width = 512
    # height = 288

    # # Generate a target in the middle of the heatmap
    # middle_target = torch.tensor([[300, 156]])  # Example ground truth

    # # Generate a target at (0, 0)
    # out_of_frame_target = torch.tensor([[0, 0]])  # Example for out-of-frame

    # # Create coordinate ranges
    # x_coords = torch.arange(width).float()
    # y_coords = torch.arange(height).float()

    # # Generate perfect Gaussian heatmaps
    # middle_x = middle_target[0, 0].float()
    # middle_y = middle_target[0, 1].float()

    # # Gaussian centered at middle_target
    # heat_map_x_perfect = torch.exp(-((x_coords - middle_x) ** 2) / (2 * (2 ** 2)))  # Sigma=2
    # heat_map_y_perfect = torch.exp(-((y_coords - middle_y) ** 2) / (2 * (2 ** 2)))  # Sigma=2

    # # Normalize heatmaps
    # heat_map_x_perfect /= heat_map_x_perfect.sum()
    # heat_map_y_perfect /= heat_map_y_perfect.sum()

    # # Reshape for batch dimension
    # heat_map_x_perfect = heat_map_x_perfect.unsqueeze(0)  # [1, W]
    # heat_map_y_perfect = heat_map_y_perfect.unsqueeze(0)  # [1, H]

    # # Visibility labels (e.g., 0: visible, 1: partially visible, 2: occluded, 3: gaussian-based)
    # visibility_values = [1]  # Example visibility values for the batch
    # visibility = torch.tensor(visibility_values).unsqueeze(1)  # Shape [B, 1]

    # # Compute the loss for a perfect match
    # loss_perfect = loss_func((heat_map_x_perfect, heat_map_y_perfect), middle_target, visibility)
    # print(f"Loss for perfect match: {loss_perfect.item()}")

    # # Compute the loss for a perfect match
    # loss_perfect = loss_func((heat_map_x_perfect, heat_map_y_perfect), middle_target, 2)
    # print(f"Loss for perfect match: {loss_perfect.item()}")

    # # Simulate predicted heatmaps and normalize using softmax
    # heat_map_x = torch.softmax(torch.randn([1, width]), dim=-1)  # Predicted x-axis probabilities
    # heat_map_y = torch.softmax(torch.randn([1, height]), dim=-1)  # Predicted y-axis probabilities

    # # Compute the loss for the middle target with noisy predictions
    # loss_middle = loss_func((heat_map_x, heat_map_y), middle_target, 1)
    # print(f"Loss for middle target with noisy predictions: {loss_middle.item()}")

    # # Compute the loss for the out-of-frame target
    # loss_out_of_frame = loss_func((heat_map_x, heat_map_y), out_of_frame_target, 0)
    # print(f"Loss for out-of-frame target: {loss_out_of_frame.item()}")

    # print(generate_gaussian_map(width=288, target_x=50, sigma=0.45))

    # --- setup ---
    H, W = 288, 512
    loss_function = HeatmapBallDetectionLoss2DWeighted(H=H, W=W, use_logits=False)  # your class

    # targets
    target_ball_position = torch.tensor([[100, 150],   # sample 0
                                        [200, 250]])  # sample 1
    visibility = torch.tensor([3, 1])  # v=3 -> Gaussian, v=1 -> one-hot

    B = target_ball_position.size(0)
    pred_map = torch.zeros(B, H*W)

    # helper: gaussian matching the loss' sigma for v=3 (here sigma=self.sigma=0.3)
    def gaussian_2d(H, W, cx, cy, sigma):
        xs = torch.arange(W).view(1, 1, W).float()
        ys = torch.arange(H).view(1, H, 1).float()
        g = torch.exp(-(((xs - cx)**2 + (ys - cy)**2) / (2.0 * sigma**2)))
        g = g / g.sum()
        return g.view(-1)  # flatten to [H*W]

    # --- make sample 0 "perfect" ---
    cx0, cy0 = target_ball_position[0, 0].item(), target_ball_position[0, 1].item()
    g0 = gaussian_2d(H, W, cx0, cy0, sigma=0.3)     # same sigma as the loss for v=3
    pred_map[0] = g0                                # match target distribution -> very low loss

    # --- make sample 1 "almost perfect" one-hot ---
    cx1, cy1 = target_ball_position[1, 0].item(), target_ball_position[1, 1].item()
    idx1 = cy1 * W + cx1
    pred_map[1].fill_(1e-6)                         # tiny elsewhere
    pred_map[1, idx1] = 1.0 - (H*W-1)*1e-6          # ~1 at GT, keeps probs in [0,1] and sums ~1 (not required for BCE)

    # compute loss
    loss = loss_function(pred_map, target_ball_position, visibility)
    print(f"Loss (one perfect Gaussian, one near one-hot): {loss.item():.6f}")

    from metrics import extract_coords2d, heatmap2d_calculate_metrics, precision_recall_f1_tracknet
    # extract predicted coordinates from the heatmap
    pred_coords = extract_coords2d(pred_map, H, W)
    print(f"Predicted coordinates for sample 0: {pred_coords}")
    # compute metrics
    metrics = heatmap2d_calculate_metrics(pred_map, target_ball_position, H, W)
    print(f"Metrics for sample 0: {metrics}")

    # compute precision, recall, f1
    precision, recall, f1, accuracy = precision_recall_f1_tracknet(pred_coords, target_ball_position)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f} (Accuracy: {accuracy:.4f})")









