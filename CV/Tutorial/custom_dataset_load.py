# 파일에서 사용자 정의 데이터셋 만들기
# 사용자 정의 Dataset 클래스는 반드시 3개함수를 구현해야 함
# __init__, __len__, and __getitem__

import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image # torch에서 이미지 읽기 위한 기능

# Dataset class를 상속받는 CustomImageDataset
class CustomImageDataset(Dataset):
    # _init__ 함수는 Dataset 객체가 생성(instantiate)될 때 한 번만 실행됩니다.
    # 여기서는 이미지와 주석 파일(annotation_file)이 포함된 디렉토리와
    # 두가지 변형(transform)을 초기화합니다.
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # __len__ 함수는 데이터셋의 샘플 개수를 반환합니다.
    def __len__(self):
        return len(self.img_labels)

    # __getitem__ 함수는 주어진 인덱스 idx 에 해당하는 샘플을 데이터셋에서 불러오고 반환합니다.
    # 인덱스를 기반으로, 디스크에서 이미지의 위치를 식별하고, read_image 를 사용하여 이미지를 텐서로 변환하고,
    # self.img_labels 의 csv 데이터로부터 해당하는 정답(label)을 가져오고,
    # (해당하는 경우) 변형(transform) 함수들을 호출한 뒤,
    # 텐서 이미지와 라벨을 Python 사전(dict)형으로 반환합니다.
    def __getitem__(self, idx):
        # iloc의 기능 : integer location의 준말로 편하게 해당 idx의 정보를 알기 위함
        # img_labels.iloc[idx, 0] :  image_label중 "idx 행"의 "0번째 열" 정보를 가져오란 뜻
        # self.img_labels는 ['file_name', 'label']로 저장되어 있으므로 0번째면 file_name 즉, 경로 반환
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        # self.img_labels는 ['file_name', 'label']로 저장되어 있으므로 1번째면 label 즉, 정답 반환
        label = self.img_labels.iloc[idx, 1]

        # 입력 전처리 적용
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        sample = {"image": image, "label": label}
        # 선택된 idx의 파일과, label dict 형태로 반환
        return sample

