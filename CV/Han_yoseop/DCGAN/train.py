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
from model import *
# import 할게 여러개라면 그냥 from util import * 로 해서 모두다 import 해도 됨
from util import save, load
from dataset import *

def train(args):
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
    nblk = args.nblk

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 결과 디렉토리 구분
    result_dir_train = os.path.join(result_dir, "train")

    # train 디렉토리 생성 코드
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir_train, "png"))

    # 데이터 전처리
    transform_train = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

    # 데이터셋 생성
    dataset_train = Dataset(data_dir=data_dir, transform=transform_train, task=task, opts=opts)

    # 데이터 로더 생성
    ## num_workers=0 으로해야 multi-process 지원 x (내 노트북 세팅에선 0으로 해야 됨)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    # 부수적인 variable 설정
    num_data_train = len(dataset_train) # train dataset 총 개수
    num_batch_train = np.ceil(num_data_train / batch_size) # train batch 개수

    # GAN task에 맞게 model, loss, optim 설정 (generator, discriminator network를 각각 설정)
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

        # 가중치 초기화
        init_weight(netG, init_type="normal", init_gain=0.02)
        init_weight(netD, init_type="normal", init_gain=0.02)

    fn_loss = nn.BCELoss().to(device) # minmax game을 위한 BCE Loss
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # 출력 Tensor -> NumPy 형태로 변환
    fn_denorm = lambda x, mean, std : (x * std) + mean # normalization 역연산해서 원래 데이터셋 형태로 복원

    st_epoch = 0
    # 중간 가중치로 이어서 학습하기
    if train_continue == "on":
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimizerG, optimD=optimizerD)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        # train 모드 키기
        netG.train()
        netD.train()

        # G,D loss 구분
        ## loss 수식에 의하면 G에는 noise z가 들어가고 D에는 input x가 들어감
        loss_G_train = [] # G(z) part
        loss_D_real_train = [] # D(x) part
        loss_D_fake_train = [] # D(G(z)) part

        for batch, data in enumerate(loader_train, 1): # dataloader를 iterate하는 것임!
            print(f"Batch {batch}: Label shape = {data['label'].shape} Label type = {data['label'].dtype}")
            # forward path
            label = data['label'].to(device)
            # input = data['input'].to(device)
            # input noise vector z 생성
            input = torch.randn(label.shape[0], 100, 1, 1).to(device)

            output = netG(input) # 생성한 이미지

            # backward path netD (근데 generator 들어갔는데 왜 D를 계산?)
            ## 우선 netD를 흐르게 해야하므로 True로 설정
            set_requires_grad(netD, True)
            optimizerD.zero_grad()

            pred_real = netD(label) # label을 넣어 판단했으므로 참이어야함
            ## 왜 detach()를 하지?
            ## 현재 루틴은 D에 대해서만 학습중 근데 output은 G에 의한 것이기 때문에 detach로 떼어내서 G의 학습은 안하고 D만 학습을 위함!
            # 즉, 온전히 D만 학습하기 위해 detach를 통해 G로 흘러가는 연결을 해제
            pred_fake = netD(output.detach()) # G(z)를 넣었을 때는 가짜로 생성한 것이므로 fake

            ## real을 real로 fake를 fake로 잘 인식하게끔 학습
            ## 왜 ones_like를 쓰지? (pred_real 차원을 갖는 모두 1인 tensor 아닌가?)
            loss_D_real = fn_loss(pred_real, torch.ones_like(pred_real))
            loss_D_fake = fn_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake) # real & fake loss의 평균 값

            loss_D.backward()
            optimizerD.step()

            # backward netG
            set_requires_grad(netD, False)
            optimizerG.zero_grad()

            ## 이때는 G를 학습하므로 detach하지 않음, D는 위에서 grad=False로 해서 학습 x
            pred_fake = netD(output)

            ## 가짜를 진짜처럼 만들어야 하므로 ones_like 활용 (?)
            loss_G = fn_loss(pred_fake, torch.ones_like(pred_fake))

            loss_G.backward()
            optimizerG.step()

            ## 손실함수 계산 (총 3개의 loss가 존재하므로 다 합쳐주기)
            loss_G_train += [loss_G.item()] # 모든 데이터에 대한 loss 누적합
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]

            print("TRAIN, EPOCH %04d / %04d | BATCH %04d / %04d |"
                  "GEN %.4f | DISC REAL: %.4f | DISC FAKE: %.4f" %
                  (epoch, num_epoch, batch, num_batch_train,
                   np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

        # 평가모드 (Generator는 평가모드 필요없으므로 제거)
        if epoch % 1 == 0: # 100 에포크마다 저장
            save(ckpt_dir, netG, netD, optimizerG, optimizerD, epoch)

def test(args):
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
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float32)]  # 유동적으로 받도록 설정

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker
    nblk = args.nblk

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 결과 디렉토리 구분
    result_dir_test = os.path.join(result_dir, "test")

    # 테스트 디렉토리 생성 코드
    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, "png"))
        os.makedirs(os.path.join(result_dir_test, "numpy"))

    # train이냐 test냐에 따라 데이터셋이나 transform, optimization, 학습 설정이 다르므로 flag로 구분하기
    transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

    dataset_test = Dataset(data_dir=data_dir, transform=transform_test, task=task, opts=opts)

    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

    # 부수적인 variable 설정
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    # GAN task에 맞게 model, loss, optim 설정 (generator, discriminator network를 각각 설정)
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

        # 가중치 초기화
        init_weight(netG, init_type="normal", init_gain=0.02)
        init_weight(netD, init_type="normal", init_gain=0.02)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    fn_denorm = lambda x, mean, std: (x * std) + mean  # normalization 역연산해서 원래 데이터셋 형태로 복원


    # test는 무조건 학습한 모델로 평가하므로 load
    netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimizerG,
                                                optimD=optimizerD)

    with torch.no_grad():
        # GAN에서 평가는 generator만 하면 됨
        # test에서는 D는 사용되지 않음
        netG.eval()
        loss_arr = []

        # GAN 평가는 1개의 노이즈 이미지를 생성만 하면 되기 때문에 for loop 필요 x
        input = torch.randn(batch_size, 1000, 1, 1).to(device)

        output = netG(input)

        # G의 output은 tanh가 거쳐진 (-1 ~ 1)사이값인데 이를 (0~1 : R/F)로 만들기 위해 denorm
        output = fn_denorm(output, mean=0.5, std=0.5)

        # batch data 개수만큼 test
        for j in range(output.shape[0]):
            id = j

            output_ = output[j]
            output_ = np.clip(output_, a_min=0, a_max=1)
            plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=None)