import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 학습 파라미터 설정
lr = 1e-6
batch_size = 4

num_epoch = 100

data_dir = './drive/MyDrive/AI_Coding/UNet/datasets'
ckpt_dir = './drive/MyDrive/AI_Coding/UNet/checkpoint'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# dataloader 구현
class Dataset(torch.utils.data.Dataset):
    # 처음 선언할 때, 할당할 argument들 설정
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform

        # dataset list에 있는 dataset들을 얻어오기
        lst_data = os.listdir(self.data_dir) # 해당 dir의 모든 파일들 list 형태로 불러오기
        # 파일들 접두사(startswith) 기반으로 data, label 구분
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        # 이렇게 정렬된 lst들을 클래스 파라미터로 설정(by self)
        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        # index에 해당하는 파일 return
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])).copy()
        input = np.load(os.path.join(self.data_dir, self.lst_input[index])).copy()

        # data가 0~255 range로 저장되어 있기 때문에 > 0 ~ 1 사이로 정규화
        label = label/255.0
        input = input/255.0

        # label에 채널 정보가 없다면 채널 축 추가
        # 채널 축은 layer 거칠수록 늘어나야하는 정보이기 때문 (학습을 위함)
        # PyTorch에 넣을거면 반드시 채널축이 있어야 됨
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # Convert to PyTorch tensors
        input = torch.from_numpy(input.transpose((2, 0, 1))).float()
        label = torch.from_numpy(label.transpose((2, 0, 1))).float()

        # 이렇게 생성된 label, input을 dict형태로 내보내기
        data = {'input' : input, 'label' : label}

        # 만약 transform을 data argument로 넣어줬다면 이걸로 적용
        #if self.transform:
        #    data = self.transform(data)

        return data


# 전처리를 위한 transform 클래스들 직접 구현
class ToTensor(object):
    # data : input과 label을 키값으로 가지는 {} 형태의 data를 object로 받음
    def __call__(self, data):
        label, input = data['label'], data['input']

        # NumPy와 PyTorch의 차원 순서는 다름
        # NumPy : (Y, X, C)
        # PyTorch : (C, Y, X)
        # NumPy to Tensor를 위한 순서 맞춤
        label = label.transpose((2, 0, 1)).copy().astype(np.float32)
        input = input.transpose((2, 0, 1)).copy().astype(np.float32)

        # data를 다시 dict 형태로 맞춰주기 (현재까지 차이는 차원 순서)
        data = {'label' : torch.from_numpy(label), 'input' : torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # 실제로 Normalization 함수 호출할 때 작동할 부분
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std # 정규화

        data = {'label' : label, 'input' : input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        # 반반확률로 flip 여부 결정
        if np.random.rand() > 0.5:
            # data 좌우반전 (fliplr)
            label = np.fliplr(label).copy() # label은 왜 flip?
            input = np.fliplr(input).copy()

        if np.random.rand() > 0.5:
            # data 상하반전
            label = np.flipud(label).copy()
            input = np.flipud(input).copy()

        data = {'label' : label, 'input' : input}

        return data

# 네트워크 학습하기 (전처리 -> 로더에 올리기 -> 학습)
transform = transforms.Compose([
    ToTensor(),
    RandomFlip(),
    Normalization(mean=0.5, std=0.5)
])

dataset_train = Dataset(os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

dataset_val = Dataset(os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)

net = UNet().to(device) # network가 학습이 되는 도메인이 gpu인지 cpu인지 명시하기 위해 to(device)

# fn_loss = nn.BCEWithLogitsLoss().to(device)
fn_loss = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# 부수적인 variable 설정
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

# output을 확인하기 위한 부수적인 function -> tensorboard로 보기 위함
# fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # 출력 Tensor -> NumPy 형태로 변환
# fn_denorm = lambda x, mean, std : (x * std) + mean # normalization 역연산해서 원래 데이터셋 형태로 복원
# fn_class = lambda x : 1.0 * (x > 0.5)

# 네트워크 학습
st_epoch = 0

# 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    torch.save({'net' : net.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_epoch%d.pth" % (ckpt_dir,epoch))

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f : int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1])) # 가장 최신 버전의 가중치 가져오기

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

# ckpt_dir 폴더자체가 없어서 이거하면 에러 발생 > 한번 학습 후 불러야..
# ckpt_lst = os.listdir(ckpt_dir)
# print(ckpt_lst[-1].split('epoch'))
# net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optimizer)

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train() # train 모드 키기
    loss_arr = []

    for batch, data in enumerate(loader_train, 1): # dataloader를 iterate하는 것임!
        print(f"Batch {batch}: Input shape = {data['input'].shape}, Label shape = {data['label'].shape}")
        # forward path
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        # backward path
        optimizer.zero_grad()
        loss = fn_loss(output, label)
        loss.backward()
        optimizer.step()

        # 손실함수 계산
        loss_arr += [loss.item()] # 모든 데이터에 대한 loss 누적합

        print("TRAIN, EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
              (epoch, num_epoch, batch, num_batch_train,np.mean(loss_arr)))

    # 평가모드
    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

    if epoch % 10 == 0: # 100 에포크마다 저장
        save(ckpt_dir, net, optimizer, epoch)
