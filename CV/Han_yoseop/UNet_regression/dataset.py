import os

import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# 관련 노이즈들 불러오기 위함
from util import *

# dataloader 와 transform 기능 구현
class Dataset(torch.utils.data.Dataset):
    # 처음 선언할 때, 할당할 argument들 설정
    def __init__(self, data_dir, transform, task=None, opts=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts

        # dataset list에 있는 dataset들을 얻어오기
        lst_data = os.listdir(self.data_dir) # 해당 dir의 모든 파일들 list 형태로 불러오기
        # BSDS500 dataset 중 .jpg, png만 다루기
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('png')]

        lst_data.sort()

        # 이렇게 정렬된 lst들을 클래스 파라미터로 설정(by self)
        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        # data가 numpy가 아니고 image이므로 image load 사용! by matplotlib
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index])).copy()

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        sz = img.shape # image size

        # 만약 세로가 가로보다 길다면
        if sz[0] > sz[1]:
            img = img.transpose((1,0,2)) # transpose해서 항상 가로로긴 이미지 반환

        if img.dtype == np.uint8:
            # data가 0~255 range로 저장되어 있기 때문에 > 0 ~ 1 사이로 정규화 (원래 uint8 일때만 255.0 으로 나눌 수 있음!)
            img = img/255.0

        # label에 채널 정보가 없다면 채널 축 추가
        # 채널 축은 layer 거칠수록 늘어나야하는 정보이기 때문 (학습을 위함)
        # PyTorch에 넣을거면 반드시 채널축이 있어야 됨
        if img.ndim == 2:
            label = img[:, :, np.newaxis]

        # 정규화된 이미지는 label로 됨
        label = img

        # task에 따라 아티팩트를 부여
        # 이를 위해 Dataset init에 task와 opts args 추가
        if self.task == "denoising":
            input = add_noise(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "inpainting":
            input = add_sampling(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "super_resolution":
            input = add_blur(img, type=self.opts[0], opts=self.opts[1])

        # 이렇게 생성된 label, input을 dict형태로 내보내기
        data = {'input' : input, 'label' : label}

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

        label = (label - self.mean) / self.std
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

class RandomCrop(object):
    # crop할 shape를 입력으로 받음
    def __init__(self, shape):
        self.shape = shape

    # input, label 모두 crop하기 (label도 사진이기 때문에)
    def __call__(self, data):
        input, label = data['input'], data['label']

        h, w = input.shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        input = input[id_y, id_x]
        label = label[id_y, id_x]

        data = {'input' : input, 'label' : label}

        return data
