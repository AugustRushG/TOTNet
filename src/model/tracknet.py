import torch.nn as nn
import torch
import torch.nn.functional as F
import time
import sys
sys.path.append('../')

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        # self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x): 

        batch_size, C, H, W = x.shape
        

        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # x = self.ups1(x)
        x = F.interpolate(x, size=(H//4, W//4), mode='nearest')
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        # x = self.ups2(x)
        x = F.interpolate(x, size=(H//2, W//2), mode='nearest')
        x = self.conv14(x)
        x = self.conv15(x)
        # x = self.ups3(x)
        x = F.interpolate(x, size=(H, W), mode='nearest')
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        # x = self.softmax(x)
        out = x.view(batch_size, self.out_channels, H, W) #[b,1,h,w]

        vertical_heatmap = out.max(dim=-1)[0].squeeze(dim=1)  # Max along width (W)
        horizontal_heatmap = out.max(dim=-2)[0].squeeze(dim=1)  # Max along height (H)

        # vertical_heatmap = torch.sigmoid(vertical_heatmap)
        # horizontal_heatmap = torch.sigmoid(horizontal_heatmap)
        vertical_heatmap = torch.softmax(vertical_heatmap, dim=-1)
        horizontal_heatmap = torch.softmax(horizontal_heatmap, dim=-1)

        return (horizontal_heatmap, vertical_heatmap), None               
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)  


class BallTrackerNetV2(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=768, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=384, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=192, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        # self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x): 

        batch_size, C, H, W = x.shape
        

        x = self.conv1(x)
        x = out1 = self.conv2(x)    
        x = self.pool1(x)
        x = self.conv3(x)
        x = out2 = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = out3 = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        # x = self.ups1(x)
        x = F.interpolate(x, size=(H//4, W//4), mode='nearest')
        concat1 = torch.concat((x, out3), dim=1)
        x = self.conv11(concat1)
        x = self.conv12(x)
        x = self.conv13(x)
        # x = self.ups2(x)
        x = F.interpolate(x, size=(H//2, W//2), mode='nearest')
        concat2 = torch.concat((x, out2), dim=1)
        x = self.conv14(concat2)
        x = self.conv15(x)
        # x = self.ups3(x)
        x = F.interpolate(x, size=(H, W), mode='nearest')
        concat3 = torch.concat((x, out1), dim=1)
        x = self.conv16(concat3)
        x = self.conv17(x)
        x = self.conv18(x)
        # x = self.softmax(x)
        out = x.view(batch_size, self.out_channels, H, W) #[B, 1, H, W]

        out = x.squeeze(dim=1).squeeze(dim=1) #[B, H, W]
        heatmap = out.view(batch_size, H*W) # Reshape to [B, H*W] for softmax
        heatmap = torch.softmax(heatmap, dim=-1)  # Apply softmax to the heatmap

        # vertical_heatmap = out.max(dim=-1)[0].squeeze(dim=1)  # Max along width (W)
        # horizontal_heatmap = out.max(dim=-2)[0].squeeze(dim=1)  # Max along height (H)

        # vertical_heatmap = torch.softmax(vertical_heatmap, dim=-1)
        # horizontal_heatmap = torch.softmax(horizontal_heatmap, dim=-1)

        return heatmap            
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)  

def extract_coords_from_heatmap(heatmap, H, W):
    """
    Extracts the (x, y) coordinates of the peak (maximum value) from a heatmap.

    Args:
    - heatmap (torch.Tensor): Heatmap tensor of shape [B, C, H * W].
    - H (int): Original height of the spatial dimensions.
    - W (int): Original width of the spatial dimensions.

    Returns:
    - coords (torch.Tensor): Tensor of shape [B, 2] containing (x, y) coordinates of the peak for each sample in the batch.
    """
    B, C, HW = heatmap.shape

    # Find the max index along the spatial dimension (H * W) for each channel
    max_spatial_indices = heatmap.argmax(dim=2)  # Shape: [B, C] (max indices within each channel's spatial map)
    max_spatial_values = heatmap.max(dim=2)[0]   # Shape: [B, C] (max values within each channel's spatial map)

    # Find the channel with the highest peak across all channels for each sample
    max_channel_indices = max_spatial_values.argmax(dim=1)  # Shape: [B] (index of channel with max value per sample)

    # Get the flattened spatial index for the highest peak in the selected channel
    flat_indices = max_spatial_indices[torch.arange(B), max_channel_indices]  # Shape: [B]

    # Convert flat indices to (x, y) coordinates using H and W
    y_coords = flat_indices // W  # Row (y) coordinate
    x_coords = flat_indices % W   # Column (x) coordinate

    # Stack x and y coordinates to get shape [B, 2]
    coords = torch.stack((x_coords, y_coords), dim=-1)  # Shape: [B, 2]
    
    return coords

def build_TrackerNet(args):
    model = BallTrackerNet(in_channels=args.num_frames*3).to(args.device)
    return model

def build_TrackNetV2(args):
    model = BallTrackerNetV2(in_channels=args.num_frames*3).to(args.device)
    return model


def create_target_heatmap(target_coords, H, W, sigma=1):
    """
    Converts target coordinates to a Gaussian heatmap.
    
    Args:
    - target_coords (torch.Tensor): [B, 2] tensor with target (x, y) coordinates.
    - H (int): Height of the heatmap.
    - W (int): Width of the heatmap.
    - sigma (float): Standard deviation for Gaussian spread.

    Returns:
    - target_heatmap (torch.Tensor): Heatmap tensor of shape [B, H, W].
    """
    B = target_coords.shape[0]
    device = target_coords.device
    target_heatmap = torch.zeros(B, H, W, device=device)

    # Generate grid of coordinates
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device)
    )
    y_grid, x_grid = y_grid.float(), x_grid.float()  # Make sure grids are float

    for i in range(B):
        x, y = target_coords[i]
        
        # Create Gaussian around target (x, y) location
        gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        target_heatmap[i] = gaussian

    
    return target_heatmap


class HeatmapCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(HeatmapCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, target):
        """
        Args:
        - predictions (torch.Tensor): Predicted heatmap of shape [B, C, H * W].
        - target (torch.Tensor): Target tensor of shape [B, H * W], containing the index of the correct class.

        Returns:
        - loss (torch.Tensor): Computed cross-entropy loss.
        """
        B, C, HW = predictions.shape
        
        # Reshape target to match the input requirements of CrossEntropyLoss
        target = target.view(B, HW).long()  # Ensure target is [B, H * W]
        
        # Apply cross-entropy loss, which expects [B, C, H * W] predictions and [B, H * W] target
        loss = self.criterion(predictions, target)
        
        return loss



    
if __name__ == '__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_occlusion_train_val_dataloader
    from losses_metrics.losses import Heatmap_Ball_Detection_Loss
    from losses_metrics.metrics import heatmap_calculate_metrics
    from model.model_utils import get_num_parameters
    from ptflops import get_model_complexity_info
    import time
    configs = parse_configs()
    configs.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs.num_frames = 5
    configs.img_size = (288, 512)

    dummy_data = torch.randn([8, 15, 288, 512])  # Dummy data for testing
    # model = build_TrackerNet(configs)
    model = build_TrackNetV2(configs)
    
    from TOTNet import benchmark_fps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {parameters/1e6:.2f} M")
    batch_data = torch.randn([5, 15, 288, 512])
    results = benchmark_fps(model, batch_data, device=device)

    print(f"Average time per pass: {results['avg_time']:.4f} s")
    print(f"Throughput: {results['fps_frames']:.2f} frames/s")
    print(f"Throughput: {results['fps_clips']:.2f} clips/s")
    