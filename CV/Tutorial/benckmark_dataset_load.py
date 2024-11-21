# torch.utils.data.DataLoader
# torch.utils.data.Dataset
# 위 2가지를 통해 벤치마크 데이터나, 커스텀 데이터를 사용하도록 함

# Dataset은 data와 label을 저장
# DataLoader는 Dataset을 data에 쉽게 접근하도록 순회 가능한 객체(iterable)로 감싸는 역할

# 데이터셋 불러오기
# root : 학습/테스트 데이터가 저장되는 경로
# train : 학습용 or 테스트용인지 여부 T/F
# download : root에 데이터가 없는 경우 인터넷에서 다운로드 여부 T/F
# transform과 target_transform은 각각 feature, label 변형(transform)을 지정

import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import datasets # 벤치마킹 데이터 불러오기 위함
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    transform=ToTensor()
)

# 데이터셋을 순회하고 시각화하기
# Dataset에 list처럼 직접 index 접근 가능
# training_data[index] 가능
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# DataLoader로 학습용 데이터 준비하기
# Dataset은 데이터셋의 특징을 가져오고 label을 지정하는 역할을 함
# 모델학습시, 샘플들을 배치단위로 전달하고 에포크마다 데이터를 shuffle하여 과적합방지

# DataLoader는 간단한 API로 위의 복잡한 과정을 추상화한 순회 가능 객체 iterable임
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# DataLoader를 통해 순회하기 (iterate)
# DataLoader에 데이터셋을 지정했다면, 데이터셋을 순회할 수 있음
# 아래의 각 iteration은 batch data에 대한 특징과 정답을 반환
# 데이터 불러오는 것을 세밀하게 제어하려면 Samplers를 활용

# 이미지와 정답 표시
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape : {train_features.size()}")
# Feature batch shape : torch.Size([64, 1, 28, 28])
print(f"Label batch shape : {train_labels.size()}")


# 64개 data중 첫번째 data에 대한 조회
# 0번째 train_feature에 대해 채널 차원(dim=0)으로 압축
# [1,28,28] -> squeeze -> [28,28]
img = train_features[0].squeeze()
label = train_labels[0]
print(f"img : {img.shape}") # [28,28]
print(f"Label: {labels_map[label.item()]}") # label이 숫자로나와서 label으로 매핑해서 클래스명 알게끔 변환
plt.imshow(img, cmap="gray")
plt.show()
