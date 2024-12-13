# parser를 위한 라이브러리
import argparse

import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 필요한 module들 import
from model import UNet
# import 할게 여러개라면 그냥 from util import * 로 해서 모두다 import 해도 됨
from util import save, load
from dataset import *

# Parser object 설정
parser = argparse.ArgumentParser(description="train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Parser에 추가할 argument들 등록하기 (등록하면 이후 터미널을 통해 자동적으로 argument들을 입력으로 넣을 수 있음)
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

# Parser에 추가할 config 설정들 등록
parser.add_argument("--data_dir", default="./datasets/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--task", default="denoising", choices=["denoising", "inpainting", "super_resolution"], type=str, dest="task")

# train or test mode 설정
parser.add_argument("--mode", default="train", type=str, dest="mode")
# 학습 처음부터 or 이어할지
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
# opts 설정 nargs를 통해 유동적인 인자 개수 받기 (opts는 어떤 기법 적용하냐에 따라 2 ~ 5개의 값을 가짐)
parser.add_argument("--opts", nargs='+' ,default=["random", 30.0], dest="opts")

# 이미지 사이즈, 채널, UNet 커널 사이즈 조정
parser.add_argument("--ny", default=320, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

# 학습 네트워크 선택 (향후 추가될 수 있으므로 []로 관리)
parser.add_argument("--network", default="unet", choices=["unet", "resnet"], type=str, dest="network")

# parser에 등록한 argument들 사용(파싱)
args = parser.parse_args() # args에는 각 parser의 arguemnt들이 들어있음

# args로 각 arguemnt들 연결
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float32)] # 유동적으로 받도록 설정

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker
network = args.network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 결과 디렉토리 구분
result_dir_train = os.path.join(result_dir, "train")
result_dir_val = os.path.join(result_dir, "val")
result_dir_test = os.path.join(result_dir, "test")

# 테스트 디렉토리 생성 코드
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, "png"))
    os.makedirs(os.path.join(result_dir_val, "png"))

    os.makedirs(os.path.join(result_dir_test, "png"))
    os.makedirs(os.path.join(result_dir_test, "numpy"))

# 네트워크 학습하기 (전처리 -> 로더에 올리기 -> 학습)
transform = transforms.Compose([
    ToTensor(),
    RandomFlip(),
    Normalization(mean=0.5, std=0.5)
])

# train이냐 test냐에 따라 데이터셋이나 transform, optimization, 학습 설정이 다르므로 flag로 구분하기
if mode == "train":
    transform_train = transforms.Compose([RandomCrop(shape=(nx,ny)) ,Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    transform_val = transforms.Compose([RandomCrop(shape=(nx, ny)),Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_train = Dataset(os.path.join(data_dir, 'train'), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_val = Dataset(os.path.join(data_dir, 'val'), transform=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)

    # 부수적인 variable 설정
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform_test = transforms.Compose([RandomCrop(shape=(nx, ny)), Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(os.path.join(data_dir, 'test'), transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

    # 부수적인 variable 설정
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

# 공통 부분 model, loss, optim 설정
net = UNet().to(device) # network가 학습이 되는 도메인이 gpu인지 cpu인지 명시하기 위해 to(device)
fn_loss = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # 출력 Tensor -> NumPy 형태로 변환
fn_denorm = lambda x, mean, std : (x * std) + mean # normalization 역연산해서 원래 데이터셋 형태로 복원
fn_class = lambda x : 1.0 * (x > 0.5)

if mode == "train":
    st_epoch = 0
    if train_continue == "on":
        # load하기
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optimizer)

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

        if epoch % 100 == 0: # 100 에포크마다 저장
            save(ckpt_dir, net, optimizer, epoch)
else:
    # eval은 무조건 학습한 모델로 평가하므로 load
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optimizer)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: | BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard로 저장하기 위해 Torch -> NumPy 형태로 변환
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            # 각각의 slice들을 따로 저장 (label, input, output)
            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                # 먼저 png형태로 파일들 저장 by plt.imsave
                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                # numpy 형태로 저장 by np.save
                # 근데 이미 label, input, output은 fn_tonumpy에 의해 numpy가 아닌가? 그냥 저장하면 안되는건가..
                # 이미 label, input, output은 numpy가 맞음 그렇기 때문에 np.save가 가능한 것
                # 다만 local 상에 .np 형태로 저장하려고 이 코드를 작성한것 뿐임
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.np' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.np' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.np' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS : %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))