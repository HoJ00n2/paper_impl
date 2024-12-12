import os
import numpy as np

import torch
import torch.nn as nn

# 네트워크 구축하기
class UNet(nn.Module):
    # init 때는 네트워크를 구현하는데 필요한 함수(layer)들을 사전에 정의해준다.
    def __init__(self):
        super().__init__()

        # 논문 구조 상 파란색 화살표에 해당하는 layer 구현
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            # 둘 다 돌아가는지 확인하기
            # 강의 상 CBR2d Layer
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)

            # 내가 짠 CBR2d Layer
            cbr = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            )

            return cbr

        # encoder 부분
        # 파란색 화살표
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(2) # 빨간색 화살표

        # 2번째 encoder
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(2)

        # 3번째 encoder
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(2)

        # 4번째 encoder
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(2)

        # 5번째 encoder
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # decoder 부분
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True) # upconv 구현
        # 4번째 decoder
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512) # 그림상 흰색 화살표에 의해 512채널이 더해짐
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # 3번째 decoder
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels= 256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # 4번째 decoder
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels= 128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # 5번째 decoder
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels= 64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)
        # output 2 -> 1로 변경 (원래 data 채널은 1이었기 때문)
        self.fc = nn.Conv2d(64, 1, 1) # 논문상 1x1 conv

    # UNet layer 연결하기 by forward
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # decoder
        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)

        cat4 = torch.cat((unpool4, enc4_2), dim=1) # concat : 채널방향으로 추가 (dim : 0 배치방향, dim : 1 채널방향, 2 : hegiht, 3 : width)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        unpool3 = self.unpool3(dec4_1)

        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        unpool2 = self.unpool2(dec3_1)

        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        unpool1 = self.unpool1(dec2_1)

        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        x = torch.sigmoid(x)

        return x # 최종 output