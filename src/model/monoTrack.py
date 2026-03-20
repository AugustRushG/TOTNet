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



class MonoTrack(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=32)
        self.conv2 = ConvBlock(in_channels=32, out_channels=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=32, out_channels=64)
        self.conv4 = ConvBlock(in_channels=64, out_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=64, out_channels=128)
        self.conv6 = ConvBlock(in_channels=128, out_channels=128)
        self.conv7 = ConvBlock(in_channels=128, out_channels=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=128, out_channels=256)
        self.conv9 = ConvBlock(in_channels=256, out_channels=256)
        self.conv10 = ConvBlock(in_channels=256, out_channels=256)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=384, out_channels=128)
        self.conv12 = ConvBlock(in_channels=128, out_channels=128)
        self.conv13 = ConvBlock(in_channels=128, out_channels=128)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=192, out_channels=64)
        self.conv15 = ConvBlock(in_channels=64, out_channels=64)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=96, out_channels=32)
        self.conv17 = ConvBlock(in_channels=32, out_channels=32)
        self.conv18 = ConvBlock(in_channels=32, out_channels=self.out_channels)

        # self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x): 

        batch_size, C, H, W = x.shape
        

        x = res1 = self.conv1(x)
        x = out1 = self.conv2(x)    

        # residual connection
        x = x + res1

        # block 2
        x = self.pool1(x)

        x = res2 = self.conv3(x)
        x = out2 = self.conv4(x)

        x = res2 + x

        x = self.pool2(x)
        x = res3 = self.conv5(x)
        x = self.conv6(x)
        x = out3 = self.conv7(x)

        x = res3 + x

        x = self.pool3(x)
        x = res4 = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = res4 + x

        # x = self.ups1(x)
        x = F.interpolate(x, size=(H//4, W//4), mode='nearest')
        concat1 = torch.concat((x, out3), dim=1)
        x = res5 = self.conv11(concat1)
        x = self.conv12(x)
        x = self.conv13(x)
        x = res5 + x
        # x = self.ups2(x)
        x = F.interpolate(x, size=(H//2, W//2), mode='nearest')
        concat2 = torch.concat((x, out2), dim=1)
        x = res6 = self.conv14(concat2)
        x = self.conv15(x)
        x = res6 + x
        # x = self.ups3(x)
        x = F.interpolate(x, size=(H, W), mode='nearest')
        concat3 = torch.concat((x, out1), dim=1)

        x = res7 = self.conv16(concat3)
        x = self.conv17(x)
        x = res7 + x
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


def build_monoTrack(args):
    model = MonoTrack(in_channels=args.num_frames*3).to(args.device)
    return model

    
if __name__ == '__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_occlusion_train_val_dataloader
    from losses_metrics.losses import Heatmap_Ball_Detection_Loss
    from losses_metrics.metrics import heatmap_calculate_metrics
    from model.model_utils import get_num_parameters
    configs = parse_configs()
    configs.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs.num_frames = 5
    configs.img_size = (288, 512)
    configs.dataset_choice = 'tennis'


    # model = build_TrackerNet(configs)
    dummy_data = torch.randn([8, 15, 288, 512])  # Dummy data for testing
    model = build_monoTrack(configs)
    
    from TOTNet import benchmark_fps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {parameters/1e6:.2f} M")
    batch_data = torch.randn([5, 15, 288, 512])
    results = benchmark_fps(model, batch_data, device=device)

    print(f"Average time per pass: {results['avg_time']:.4f} s")
    print(f"Throughput: {results['fps_frames']:.2f} frames/s")
    print(f"Throughput: {results['fps_clips']:.2f} clips/s")

