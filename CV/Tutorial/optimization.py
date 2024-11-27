# 모델 매개변수 최적화하기

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data,64,True)
test_dataloader = DataLoader(test_data, 64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# 하이퍼 파라미터
learning_rate = 1e-3
batch_size = 64
epochs = 5

# 손실함수
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 최적화 수행 (train_loop, test_loop)
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 모델을 학습 모드로 설정 : bn, dropout layer 흐르도록 설정
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        loss.backward() # step 1
        optimizer.step() # step 2
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss : {loss:>7f} [{current:5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # 모델을 평가모드로 설정 : bn, dropout layer 적용 안하도록
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # torch.no_grad()를 통해 gradient 흐르지 않도록 (학습 방지)
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% Avg loss: {test_loss:>8f}\n")

epochs = 10
for e in range(epochs):
    print(f"Epoch {e+1}\n ------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")