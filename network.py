import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, k=3, s=1, p=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_feat, out_feat, k, s, p, bias=False
            ),
            nn.BatchNorm2d(out_feat),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.body(x)
        return x

class modelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = nn.Sequential(
            ConvBlock(3, 32, 3, 2, 1),
            ConvBlock(32, 64, 3, 2, 1),
            ConvBlock(64, 128, 3, 2, 1),
            ConvBlock(128, 128, 3, 2, 1),
            nn.AdaptiveMaxPool2d(1)
        )
        self.head = nn.Linear(
            128, 10
        )
    
    def forward(self, x):
        x = self.fe(x).squeeze()
        x = self.head(x)
        return x

# Depthwise Convolution 구현
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

# Pointwise Convolution 구현
class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pointwise(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class DSBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.body = nn.Sequential(
            DepthwiseConv(in_feat, 3),
            PointwiseConv(in_feat, out_feat),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        return self.body(x)

class modelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = nn.Sequential(
            DSBlock(3, 32),
            DSBlock(32, 64),
            DSBlock(64, 128),
            DSBlock(128, 256),
            nn.Flatten()
        )
    
        self.head = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fe(x)
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    # TEST modelA
    net = modelB()
    random_batched_input = torch.randn(16, 3, 28, 28)
    _ = net(random_batched_input)