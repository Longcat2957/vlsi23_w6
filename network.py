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
        self.fe = nn.Sequential()
        self.head = nn.Linear()
    
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


class modelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = nn.Sequential(

        )
    
        self.head = None
    
    def forward(self, x):
        x = self.fe(x)
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    # TEST modelA
    net = modelB()
    random_batched_input = torch.randn(16, 3, 28, 28)
    _ = net(random_batched_input)