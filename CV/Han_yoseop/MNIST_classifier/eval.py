## 필요한 라이브러리 설치
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision import transforms, datasets

## 트레이닝 파라미터 설정
lr = 1e-3
batch_size = 64
num_epoch = 10

ckpt_dir = "./checkpoint"
log_dir = "./log" # 텐서보드 파일들이 저장될 곳

device = torch.device("cpu")

## 네트워크 구축
class Net(nn.Module):

    def __init__(self):
        # super(Net, self).__init__()
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

## 네트워크를 save, load
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

## MNIST dataset 불러오기
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
    ])

dataset = datasets.MNIST(train=False, transform=transform, root='./')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_data = len(loader.dataset) # 총 데이터 개수 근데 왜 loader.dataset으로 하지?

num_batch = np.ceil(num_data/batch_size) # 60000 / 64 = 937.5의 ceil(올림) >> 938개의 배치

# 네트워크와 손실함수 설정
model = Net()
params = model.parameters()

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda output: torch.softmax(output, dim=1) # 최종 결정
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

optim = torch.optim.Adam(params, lr=lr)
writer = SummaryWriter(log_dir=log_dir)

# 학습 가중치 load
model, optim = load(ckpt_dir=ckpt_dir, net=model, optim=optim)

## 학습을 위한 for문

# 요섭.ver
with torch.no_grad():
    model.eval()

    loss_arr = []
    acc_arr = []

    for batch, (input, label) in enumerate(loader, 1):
        input = input.to(device)
        label = label.to(device)

        output = model(input) # loss를 위한 logit
        y_pred = fn_pred(output) # acc를 위한 예측 label

        # loss 구하기
        # 실수로 loss = fn_loss(output, y_pred) 하니까 loss가 폭발함
        loss = fn_loss(output, label)
        acc = fn_acc(y_pred, label)

        # 평가 땐 이 부분들도 필요 없음
        # optim.zero_grad()
        # loss.backward()
        # optim.step()

        loss_arr += [loss.item()]
        acc_arr += [acc.item()]

        print('Test: Batch: %04d/%04d | Loss: %.4f | Acc : %.4f' %
              (batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))

# 파이토치 튜토리얼.ver
# for t in range(num_epoch):
#     size = len(loader.dataset)
#
#     model.train()
#     for batch, (X, y) in enumerate(loader):
#         y_pred = model(X)
#
#         # loss
#         loss = fn_loss(y_pred, y)
#
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#
#         loss = loss.item()
#         print(f"loss : {loss:>7f}")
