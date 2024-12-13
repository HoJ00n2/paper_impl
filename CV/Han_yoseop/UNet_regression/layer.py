import torch
import torch.nn as nn

# 이 구조는 여러 네트워크에서 기본적으로 사용하는 layer 이므로 따로 빼둠
# 논문 구조 상 파란색 화살표에 해당하는 layer 구현 (conv + batchnorm + relu 세트)

# 원래는 함수로 정이했는데, nn.module을 상속받기 위해 class로 구현
class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None:
            layers += [nn.ReLU() if relu==0.0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)
    def forward(self, x):
        return self.cbr(x)