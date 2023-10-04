import torch
import torch.nn as nn
import thop

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



if __name__ == "__main__":
    # 테스트를 위한 입력 데이터 생성
    input = torch.randn(1, 3, 224, 224)  # 입력 이미지 크기는 (배치 크기, 채널 수, 높이, 너비)로 가정

    # 일반적인 컨볼루셔널 레이어 생성
    conv_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
    out = conv_layer(input)

    # Depthwise Convolution 생성
    depthwise_conv = DepthwiseConv(3, kernel_size=3)

    # Pointwise Convolution 생성
    pointwise_conv = PointwiseConv(3, 64)


    # 연산량 및 파라미터 수 계산
    flops_conv, params_conv = thop.profile(conv_layer, inputs=(input,))
    flops_depthwise, params_depthwise = thop.profile(depthwise_conv, inputs=(input,))
    flops_pointwise, params_pointwise = thop.profile(pointwise_conv, inputs=(input,))

    print("Convolution 연산량 (GFLOPs):", flops_conv)
    print("Convolution 파라미터 수 (Millions):", params_conv)
    print("Depthwise Convolution 연산량 (GFLOPs):", flops_depthwise)
    print("Depthwise Convolution 파라미터 수 (Millions):", params_depthwise)
    print("Pointwise Convolution 연산량 (GFLOPs):", flops_pointwise)
    print("Pointwise Convolution 파라미터 수 (Millions):", params_pointwise)