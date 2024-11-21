import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tesnor(y),value=1))
)

# 코드 동작 예시
y = 4
one_hot = torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1) # index : tensor type으로 넣어야해서 torch.tensor(y)로 넣은 것
print(torch.tensor(y)) # tensor(4)
print(one_hot) # tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])

# 위의 과정을 lambda로 custom 특정 변환을 수행 -> 결국 one-hot encoding으로 만들기 위함
