import os

import numpy as np

import torch
import torch.nn as nn

# dataloader 와 transform 기능 구현
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