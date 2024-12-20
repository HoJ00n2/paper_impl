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

        self.to_tensor = ToTensor()
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

        # 이렇게 생성된 label을 dict형태로 내보내기
        # 왜 data에 input 없애고, input 받아주는 쪽에 img 대신 data['label']로 받아온거지?
        # 대신 노이즈 아티팩트를 부여할 때 data에 input이란 key로 받아서 저장함
        # 아마 input (노이즈 이미지), label (원래 깨끗한 이미지)를 구분하기 위함인 듯
        # 기존 데이터는 원본데이터만 주어졌고, 노이즈 데이터가 따로 주어지지 않았기에 우리가 노이즈를 구현해서 부여하는식
        data = {'label' : label}

        # task에 따라 아티팩트를 부여
        # 이를 위해 Dataset init에 task와 opts args 추가
        if self.task == "denoising":
            data['input'] = add_noise(data['label'], type=self.opts[0], opts=self.opts[1])
        elif self.task == "inpainting":
            data['input'] = add_sampling(data['label'], type=self.opts[0], opts=self.opts[1])

        if self.transform:
            data = self.transform(data)

        # sr은 input, ouput resolution이 다르므로 crop 같은걸 할 때, 에러 발생 위험
        # 먼저 전처리를 하고 이후 sr을 적용하도록 수정
        if self.task == "super_resolution":
            data['input'] = add_blur(data['label'], type=self.opts[0], opts=self.opts[1])

        data = self.to_tensor(data)

        return data


# 전처리를 위한 transform 클래스들 직접 구현
class ToTensor(object):
    # data : input과 label을 키값으로 가지는 {} 형태의 data를 object로 받음
    def __call__(self, data):
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).copy().astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data

class RandomFlip(object):
    def __call__(self, data):
        # 반반확률로 flip 여부 결정
        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        return data

class RandomCrop(object):
    # crop할 shape를 입력으로 받음
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        label = data['label']

        h, w = label.shape[:2] # input.shape의 0번째 1번째 요소 반환
        new_h, new_w = self.shape # shape는 내가 train에서 지정한 nx,ny = (480, 320)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data
