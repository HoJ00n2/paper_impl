import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = ("cpu")

# 클래스 정의하기
# 신경망 모델을 nn.Module의 하위클래스로 정의
# __init__ 에서 신경망 계층들을 초기화
# nn.Module을 상속받는 모든 클래스들은 forward 메소드에 입력데이터에 대한 연산을 구현함
class NeuralNetwork(nn.Module):
    def __init__(self):
        # nn.Module이라는 부모클래스로 초기화하기 위함
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1,28,28, device=device) # tensor(1x28x28) data device에 올리기
logits = model(X)
# 모델 입력의 결과인 logits는 2차원 텐서를 반환
# 모델 출력의 dim=0은 각 class에 대한 예측값 10개
# dim=1에는 각 출력의 개별 값
# print(f"logits : {logits}")
# print(f"logits.shape : {logits.shape}") # [1,10] 10개 class에 대한 예측값

# nn.Softmax(dim=1)의 return은 Softmax 모듈의 인스턴스 생성
# 이 인스턴스에 대한 입력으로 (logits)를 넣어준 것
# nn.Softmax말고 F.softmax(logits, dim=1)을 쓸 수 있음

pred_prob = nn.Softmax(dim=1)(logits)
print(f"pred_prob: {pred_prob}") # pred_prob: tensor([[0.0943, 0.1005, 0.0992, 0.0948, 0.1094, 0.0916, 0.0988, 0.1081, 0.1041,0.0993]])
y_pred = pred_prob.argmax(1) # argmax(dim) dim 기준 최대값의 idx 반환, dim=1이므로 열방향중 최대 값을 가지는 idx
print(f"Predicted class: {y_pred}") # tensor([4])

print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}\n")

