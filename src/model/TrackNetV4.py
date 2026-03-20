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
    
class MotionAttention(nn.Module):
    """
    Builds motion attention maps A_t from absolute frame differencing with
    learnable power normalization, then reduces/aggregates them to a few channels.
    """
    def __init__(self, in_channels=3, reduce_to=1, use_grayscale=True):
        super().__init__()
        self.use_grayscale = use_grayscale
        self.__in_channels = in_channels
        # Learnable power-normalization: y = ((x + eps)^gamma - eps) * alpha + beta, then clamp/sigmoid
        # You can also try y = sigmoid(alpha * (x ** gamma) + beta)
        self.gamma = nn.Parameter(torch.tensor(0.5))  # start < 1 to emphasize small motions
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
        self.eps   = 1e-6

        # Reduce (T-1) differencing maps to `reduce_to` channels via 1x1 conv after a small CNN
        # We first process each D^+ frame with a tiny conv block shared across time.
        self.per_map = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # After stacking along channel dim, squeeze to `reduce_to` channels
        self.reducer = nn.Conv2d(8, reduce_to, kernel_size=1)

        # Optional: learnable weights to aggregate across time (T-1)
        self.time_weight = None  # set dynamically once T is known

    def rgb_to_gray(self, x):
        # x: [B,T,C,H,W] in RGB
        r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b  # [B,T,H,W]

    def power_norm(self, d_abs):
        # d_abs in [0, +inf) ideally normalized to [0,1]
        y = ((d_abs + self.eps) ** torch.clamp(self.gamma, 0.05, 5.0))
        y = self.alpha * y + self.beta
        return torch.sigmoid(y)  # maps to [0,1]

    def forward(self, x):
        """
        x: [B,T*C,H,W] in float
        returns:
           A: motion attention map [B, A_ch, H, W] with A_ch = reduce_to
        """
        B, T_C, H, W = x.shape
        T = T_C // 3  # number of frames
        x = x.view(B, T, 3, H, W)  # [B,T,C,H,W]
        if self.use_grayscale:
            xg = self.rgb_to_gray(x).unsqueeze(2)  # [B,T,1,H,W]
        else:
            # If not grayscale, you can convert via a 1x1 conv; grayscale is simpler/robust
            xg = x.mean(dim=2, keepdim=True)      # [B,T,1,H,W]

        # Absolute frame differencing: D^+_t = |F_{t+1} - F_t|
        d = torch.abs(xg[:, 1:] - xg[:, :-1])     # [B,T-1,1,H,W]

        # Normalize to [0,1] per-sample (robust): optional but helpful
        d = d / (d.amax(dim=(2,3,4), keepdim=True) + 1e-6)

        # Power normalization (learnable) -> attention per timestep
        a_list = []
        for t in range(d.size(1)):
            dt = d[:, t]                        # [B,1,H,W]
            at = self.power_norm(dt)            # [B,1,H,W]
            # ft = self.per_map(at)               # [B,8,H,W]
            a_list.append(at)

        # Simple aggregation across time: average (or learn weights)
        a_stack = torch.stack(a_list, dim=1)     # [B,T-1,C,H,W]
        a_stack = a_stack.view(B, (T-1), H, W)  # [B,T-1,C,H,W]

        return a_stack


class TrackNetV4(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_frames = in_channels // 3  # Assuming RGB input, e.g., 9 channels for 3 frames
        self.motion_attention = MotionAttention(in_channels=in_channels, reduce_to=1, use_grayscale=True)

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
        self.conv17 = ConvBlock(in_channels=64+(self.num_frames-1), out_channels=64+(self.num_frames-1))
        self.conv18 = ConvBlock(in_channels=64+(self.num_frames-1), out_channels=self.out_channels)

        # self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x): 

        batch_size, C, H, W = x.shape
        
        motion_promot = self.motion_attention(x)  # [B, T-1, 1, H, W]


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
        x = torch.cat((x, motion_promot), dim=1)  # Concatenate motion attention maps
        x = self.conv17(x) # [batch_size, 64, H, W]
        x = self.conv18(x)
        # x = self.softmax(x)
        out = x.view(batch_size, self.out_channels, H, W) #[B, 1, H, W]

        out = x.squeeze(dim=1).squeeze(dim=1) #[B, H, W]
        heatmap = out.view(batch_size, H*W) # Reshape to [B, H*W] for softmax
        heatmap = torch.softmax(heatmap, dim=-1)  # Apply softmax to the heatmap


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



if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    from TOTNet import benchmark_fps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrackNetV4(in_channels=15, out_channels=1).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {parameters/1e6:.2f} M")
    batch_data = torch.randn([5, 15, 288, 512])
    results = benchmark_fps(model, batch_data, device=device)

    print(f"Average time per pass: {results['avg_time']:.4f} s")
    print(f"Throughput: {results['fps_frames']:.2f} frames/s")
    print(f"Throughput: {results['fps_clips']:.2f} clips/s")
    