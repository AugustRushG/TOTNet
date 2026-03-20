import torch
import torch.nn as nn
import sys
import time
from einops import rearrange

sys.path.append('../')

    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)
    


class EncoderBlock(nn.Module):
    def __init__(self, pool_size, in_channels, out_channels, spatial_kernel_size, temporal_kernel_size, 
                 padding='same', spatial_padding='same', bias=True, num_spatial_layers=2, num_temporal_layers=1):
        super().__init__()
        self.out_channels = out_channels

        self.conv_layers = nn.ModuleList()
        self.temp_layers = nn.ModuleList()
        for i in range(num_spatial_layers):
            self.conv_layers.append(ConvBlock(
                in_channels=in_channels if i == 0 else out_channels,  # Input channels for the first layer
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                pad=spatial_padding
            ))
        
        for i in range(num_temporal_layers):
            self.temp_layers.append(TemporalConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=temporal_kernel_size,
                padding=padding,
                bias=bias
            ))
       
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3d = nn.AdaptiveMaxPool3d(pool_size)

    def forward(self, x, num_frames):
        # input in shape [BN, C, H, W]
        BN, C, H, W = x.shape
        B = BN//num_frames
  
        for layer in self.conv_layers:
            x = layer(x)

        spatial_out = x.clone()
        x = rearrange(x, "(b n) c h w -> b c n h w", b=B, n=num_frames)  # [B, C', N, H, W]
        x_res = x  # Residual connection

        # Temporal Convolution using Conv3d
        for layer in self.temp_layers:
            x_temporal = layer(x)
       
        temporal_out = x_temporal.clone()
        x = x_temporal + x_res  # Add residual
     
        x = self.pool3d(x)
        _, _, N, _, _ = x.shape
       
        # reshape to [B*N, C, H, W]
        x = rearrange(x, 'b c n h w -> (b n) c h w')

        return x, spatial_out, temporal_out, N

class DecoderBlock(nn.Module):
    def __init__(self, up_size, in_channels, out_channels, spatial_kernel_size, temporal_kernel_size, 
                padding='same', spataial_padding='same', bias=True, final=False, num_spatial_layers=2, num_temporal_layers=1):
        super().__init__()
        self.out_channels = out_channels
        self.final = final
        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.Upsample(size=up_size, mode='trilinear')

        self.conv_layers = nn.ModuleList()
        self.temp_layers = nn.ModuleList()

        for i in range(num_spatial_layers):
            self.conv_layers.append(ConvBlock(
                in_channels=in_channels if i == 0 else out_channels,  # Input channels for the first layer
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                pad=spataial_padding
            ))
        
        for i in range(num_temporal_layers):
            self.temp_layers.append(TemporalConvBlock(
                in_channels=out_channels*2 if i ==0 else out_channels,
                out_channels=out_channels,
                kernel_size=temporal_kernel_size,
                padding=padding,
                bias=bias
            ))
        
        if final == True:
            self.residual_proj = TemporalConvBlock(in_channels=out_channels, out_channels=1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
            self.temp_layers.append(TemporalConvBlock(out_channels, 1, temporal_kernel_size, padding, bias))


    def forward(self, x, spatial_concat, temporal_concat):
        # input in shape [BN, C, H, W]
        x = self.up(x)
        B, C, N, H, W = x.shape
        x = rearrange(x, 'b c n h w -> (b n) c h w', b=B, n=N)
        x = torch.concat((x, spatial_concat), dim=1)

        for layer in self.conv_layers:
            x = layer(x)

        x = rearrange(x, '(b n) c h w -> b c n h w', b=B, n=N)  # [B, C', N, H, W]
        x_res = x
        # Temporal Convolution using Conv3d
        x = torch.concat((x, temporal_concat), dim=1)

        for layer in self.temp_layers:
            x = layer(x)

        if self.final:
            x_res = self.residual_proj(x_res)  # Project to [B, 1, N, H, W]
        
        x = x + x_res
        # x = rearrange(x, 'b c n h w -> (b n) c h w')

        return x


class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernel_size, temporal_kernel_size, 
                 padding='same', bias=True, num_spatial_layers=2, num_temporal_layers=1):
        super().__init__()
        self.out_channels = out_channels

        self.conv_layers = nn.ModuleList()
        self.temp_layers = nn.ModuleList()

        for i in range(num_spatial_layers):
            self.conv_layers.append(ConvBlock(
                in_channels=in_channels if i == 0 else out_channels,  # Input channels for the first layer
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                pad=padding
            ))
        
        for i in range(num_temporal_layers):
            self.temp_layers.append(TemporalConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=temporal_kernel_size,
                padding=padding,
                bias=bias
            ))

    def forward(self, x, N):
        # Block 4 which is the bottleneck block
        BN, C, H, W = x.shape
        B = BN//N

        for layer in self.conv_layers:
            x = layer(x)

        x = rearrange(x, '(b n) c h w -> b c n h w',b=B, n=N)
        x_res = x  # Residual connection

        # Temporal Convolution using Conv3d
        x_temporal = None
        for layer in self.temp_layers:
            x_temporal = layer(x)
        
        if x_temporal != None:
            x = x_temporal + x_res  # Add residual
        else:
            x = x_res

        # x = rearrange(x, 'b c n h w -> b c n h w', b=B, n=self.num_frames)

        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(ClassificationHead, self).__init__()
        
        # Create a list of layers
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))  # Linear layer
            layers.append(nn.ReLU())                     # Activation
            layers.append(nn.Dropout(dropout))               # Optional dropout
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, output_dim))  # Linear layer for output classes
        
        # Combine layers into a Sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TemporalConvNet(nn.Module):
    def __init__(self, input_shape=(288, 512), spatial_channels=64, num_frames=5):
        super(TemporalConvNet, self).__init__()

        self.spatial_channels = spatial_channels
        self.num_frames = num_frames
        self.convblock1_out_channels = spatial_channels * 2
        self.convblock2_out_channels = spatial_channels * 4
        self.convblock3_out_channels = spatial_channels * 8
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=-1)

        size = (num_frames, input_shape[0], input_shape[1])  # Keep original shape
        size1 = (5, input_shape[0] // 2, input_shape[1] // 2)  # Integer division
        size2 = (3, input_shape[0] // 4, input_shape[1] // 4)  # Integer division
        size3 = (1, input_shape[0] // 8, input_shape[1] // 8)  # Integer division


        # block 1
        # Spatial convolutions
        self.block1 = EncoderBlock(pool_size=size1, in_channels=3, out_channels=spatial_channels, 
                                spatial_kernel_size=3, temporal_kernel_size=(size[0], 3, 3))

        # block 2 
        self.block2 = EncoderBlock(pool_size=size2, in_channels=spatial_channels, out_channels=self.convblock1_out_channels, 
                                   spatial_kernel_size=2, temporal_kernel_size=(3, 2, 2), 
                                   num_spatial_layers=2, num_temporal_layers=2)

        #block 3
        self.block3 = EncoderBlock(pool_size=size3, in_channels=self.convblock1_out_channels, out_channels=self.convblock2_out_channels, 
                                   spatial_kernel_size=2, temporal_kernel_size=(3, 2, 2), 
                                   num_spatial_layers=2, num_temporal_layers=2)

        self.bottle_neck = BottleNeckBlock(in_channels=self.convblock2_out_channels, out_channels=self.convblock3_out_channels,
                                           spatial_kernel_size=1, temporal_kernel_size=(1, 1, 1), 
                                           num_spatial_layers=3, num_temporal_layers=2)

        #block 5
        self.block5 = DecoderBlock(size2, self.convblock3_out_channels+self.convblock2_out_channels, self.convblock2_out_channels, 
                                   spatial_kernel_size=2, temporal_kernel_size=(3, 2, 2), 
                                   num_spatial_layers=2, num_temporal_layers=2)

        #block 6
        self.block6 = DecoderBlock(size1, self.convblock2_out_channels+self.convblock1_out_channels, self.convblock1_out_channels, 
                                   spatial_kernel_size=2, temporal_kernel_size=(3, 2, 2), 
                                   num_spatial_layers=2, num_temporal_layers=2)

        #block 7
        self.block7 = DecoderBlock(size, self.convblock1_out_channels+self.spatial_channels, self.spatial_channels, 
                                   spatial_kernel_size=3, temporal_kernel_size=(size[0], 3, 3), final=False)


        # projection block
        self.temp_reduce = TemporalConvBlock(in_channels=self.spatial_channels, out_channels=1, kernel_size=(num_frames, 1, 1), padding=(0, 0, 0))
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0) 
            elif isinstance(module, nn.Linear):
                # Initialize linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, C, H, W]
        Returns:
            tuple: heatmap in both x and y directions 
        """
        B, N, C, H, W = x.shape
        
        # Reshape to [B*N, C, H, W] for spatial convolutions
        x = rearrange(x, 'b n c h w -> (b n) c h w', b=B, n=N) # Merge batch and frame dimensions

        # Block 1
        x, spatial_out1, temporal_out1, N = self.block1(x, N)

        # Block 2
        x, spatial_out2, temporal_out2, N = self.block2(x, N)

        # Block 3
        x, spatial_out3, temporal_out3, N = self.block3(x, N)

        # block 4 bottleneck
        x = self.bottle_neck(x, N)
       
        # block 5
        x= self.block5(x, spatial_out3, temporal_out3)
        
        # block 6
        x = self.block6(x, spatial_out2, temporal_out2)

        # block 7
        x = self.block7(x, spatial_out1, temporal_out1) #outputs [B*N, C, H, W] 

        x = self.temp_reduce(x) 
        out = x.squeeze(dim=1).squeeze(dim=1) #[B, H, W]
        heatmap = out.view(B, H*W) # Reshape to [B, H*W] for softmax
        heatmap = self.softmax(heatmap)  # Apply softmax to the heatmap
        # heatmap = heatmap.view(B, H, W)  # Reshape back to [B, H, W]

        return heatmap



def build_motion_model_light(args):
    # motion_model = MotionModel()
    model = TemporalConvNet(input_shape=(288, 512), spatial_channels=args.num_channels, num_frames=args.num_frames).to(args.device)
    return model


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
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    total_time = 0.0
    for _ in range(num_iters):
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            _ = model(batch_data)

        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_time = total_time / num_iters  # seconds per forward pass

    # Throughput calculations
    if batch_data.ndim == 4:   # [B, C*T, H, W] = images
        num_frames = batch_data.shape[0]
        num_clips = num_frames/3
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


if __name__ == '__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_occlusion_train_val_dataloader
    from model.model_utils import get_num_parameters
    from ptflops import get_model_complexity_info

    configs = parse_configs()
    configs.num_frames = 5
    configs.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs.batch_size = 5
    configs.img_size = (288, 512)
    configs.dataset_choice = 'tennis'
    configs.num_channels = 32

    batch_data = torch.randn([5, 5, 3, 288, 512])
    
    motion_model = build_motion_model_light(configs)
    print(f"motion model num params is {get_num_parameters(motion_model)}")
    
    results = benchmark_fps(motion_model, batch_data, device=configs.device)

    print(f"Average time per pass: {results['avg_time']:.4f} s")
    print(f"Throughput: {results['fps_frames']:.2f} frames/s")
    print(f"Throughput: {results['fps_clips']:.2f} clips/s")


