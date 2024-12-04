import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

# 트레이닝 파라미터 설정
lr = 1e-3
batch_size = 64
num_epoch = 10

ckpt_dir = "./checkpoint"
log_dir = "./log" # 텐서보드 파일들이 저장될 곳

device = torch.device("cpu")

# 네트워크 구축
class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,10, 5,1,0, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(10,20,5,1,0,bias=True)
        self.drop = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(320, 50, bias=True)
        self.fc2 = nn.Linear(50,10, bias=True)

    def forward(self, x):
        # x : (1, 28, 28) MNIST data
        x = self.conv1(x) # (10, 24, 24)
        x = self.pool(x) # (10, 12, 12)
        x = self.relu(x)

        x = self.conv2(x) # (20, 8, 8)
        x = self.drop(x)
        x = self.pool(x) # (20, 4, 4)
        x = self.relu(x)

        x = x.view(-1, 320) # 이후 fc1으로 계산을 위해 차원 변환 (20, 4, 4) -> (1, 320)
        x = self.fc1(x) # (1, 50)
        x = self.relu(x)
        x = self.drop(x)

        x = self.fc2(x) # (1, 10)

        return x

# 네트워크를 save, load
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.mkdir(f"./{ckpt_dir}")

        torch.save({'net' : net.state_dict(), 'optim' : optim.state_dict()},
                   './%s/model_epoch%d.pth' % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    # torch.save로 저장한 키값 불러오기
    # network parameter(weight & bias)와 optimzer parameter의 가중치들 적용
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim

