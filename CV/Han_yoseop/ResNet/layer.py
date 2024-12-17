import torch
import torch.nn as nn

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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnrom", relu=0.0):
        super().__init__()

        layers = []

        # 첫번째 CBR2d
        layers += [CBR2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=bias, norm=norm, relu=relu)]
        # 2번째 CBR2d (activation은 없는 듯함)
        layers += [CBR2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=bias, norm=norm, relu=None)]

        self.resblk = nn.Sequential(*layers) # 1,2번째 CBR2d 이어붙인 resblk

    def forward(self, x):
        return x + self.resblk(x) # resblok 마지막 부분의 elment sum

class PixelUnShuffle(nn.Module):
    def __init__(self, rx=2, ry=2):
        super().__init__()
        self.rx = rx # self를 하는 것은 class 멤버변수로 만드는 과정
        self.ry = ry

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        # x input을 reshape (B,C는 냅두고 H,W를 downsampling)
        # 즉, x.reshape(target) 중 target이 x가 변하고자 하는 차원 형태로, 이 때 되고자 하는 shape가 tuple 형태로 전달됨
        # H는 곧 (H // ry) * ry 이니까 저렇게 쪼갤 수 있는건 이해가 가는데 왜 쪼개고 차원을 늘려서 관리할까?
        # 이 기법은 블록 기반 작업이라고도 불림 (이미지를 작은 패치나 블록단위로 쪼개서 처리)
        x = x.reshape(B, C, H // ry, ry, W // rx, rx) # axis를 4개에서 6개로 증가

        # 증가된 axis에 대해서 새로 재배열 by permute
        x = x.permute(0, 1, 3, 5, 2, 4) # (B, C, ry, rx, H//ry, W//rx)의 순서로 재배열

        # 다시 reshape, 이렇게 함으로써 위의 ry, rx axis가 C axis로 통합됨 (총 dim은 유지)
        x = x.reshape(B, C * ry * rx, H // ry, W // rx)

        return x

class PixelShuffle(nn.Module):
    def __init__(self, rx=2, ry=2):
        super().__init__()
        self.rx = rx
        self.ry = ry

    def forward(self, x):
        # PixelShuffle의 역순으로 실행해주면 됨
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)

        x = x.permute(0, 1, 4, 2, 5, 3) # shuffle의 첫번째 reshape dim과 일치시켜주기 (B, C, H, ry, W, rx)

        x = x.reshape(B, C // (ry * rx), H * ry, W * rx) # 이렇게 하면 고해상도의 이미지 생성

        return x
