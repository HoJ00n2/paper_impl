import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.128,0.456),
    ])
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.128,0.456)
    ])
)

batch_size = 64

# create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [B, C, H, W] : {X.shape}") # Shape of X [B, C, H, W] : torch.Size([64, 1, 28, 28])
    print(f"Shape of y : {y.shape} {y.dtype}") # Shape of y : torch.Size([64]) torch.int64
    break

# Get CPU for training
device = ("cuda" if torch.cuda.is_available() else "cpu")

# define Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # nn.Flatten(start_dim=1, end_dim=-1) 로 1번째 부터 -1까지의 요소를 모두 concat 시킴
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # print(x.shape) # [1,28,28] # 1C x 28H x 28W
        x = self.flatten(x)
        # print(f"flattend x.shape : {x.shape}") # 1x28x28 이미지를 flatten 시켰으므로 >> [1,784]
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model) # forward에서 구성한 내용이 model의 구조가 됨
             # forward에서 처음에 self.flatten(x)로 시작하고 다음엔, self.linear_relu_stack(x) layer가 쌓여서 아래의 결과가 출력
# NeuralNetwork(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#   ))

# Optimizing the Model Parameters
# todo CrossEntropyLoss, SGD 한번 구현해보기
loss_fn = nn.CrossEntropyLoss() # 여기서 init에 필요한애들 설정
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # 연산을 위해 올리기
        pred = model(X) # 데이터 넣고 추론
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # 중복 계산 방지를 위해 초기화

        if batch % 100 == 0:
            # 100번째 batch 당시의 loss값 출력,
            # loss의 scalar 값으로 받기 위해 loss.item() 으로 받아오기
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f} [{current:5d}/{size:5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader) # dataloader엔 이미 batch단위로 담기기에 총 몇번의 batch를 실행했는가?
    model.eval() # 학습, 평가 때 전략을 다르게 적용하기 위함 (bn, dropout)
    test_loss, correct = 0, 0
    with torch.no_grad(): # 기울기 자체를 반영하지 않기 위함
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print(f"test_pred: {pred.shape}") # 64개 batch에 대해 10가지 확률을 내놓으므로 ([64, 10])의 shape를 가짐
            # loss.item()이 실질적인 loss 값
            test_loss += loss_fn(pred, y).item() # 얘는 왜 누적하는거지? -> 이후 num_batches로 나누어 평균 loss 계산
            # 배치마다 각 예측값과 실제값이 맞는 경우 float type으로 하여 총합을 반환
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # test_loss는 각 데이터가 아닌 1개 batch에 대한 loss 이므로 batch 개수만큼 나눔
    test_loss /= num_batches # 모든 데이터에 대한 total loss 계산
    # correct는 각 데이터에 대한 맞은 개수를 찾으므로 data 총 개수인 size로 나눔
    correct /= size # 모든 데이터에 대한 맞은 개수 계산
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% Avg loss: {test_loss:>8f}\n")

# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n -------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

# Saving Models
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

# The process for loading a model includes re-creating the model structure and loading the state dictionary into it.
# # load Models
# model = NeuralNetwork().to(device) # Network 초기화하여 device에 올리기
# model.load_state_dict(torch.load("model.pth", weights_only=True)) # Network에 학습 가중치 끼우기

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)

    # print(f"pred : {pred}, {pred.shape}")
    # pred : tensor([[-0.9224, -1.5964, -1.6050, -0.7822, -0.0090, -0.0674, -2.6987,  5.0267,
    #          -0.7143,  1.9831]]), torch.Size([1, 10])
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    # print(f" {pred[0]}, {pred[0].argmax()}") # tensor([-0.9224, -1.5964, -1.6050, -0.7822, -0.0090, -0.0674, -2.6987,  5.0267,
                                             # -0.7143,  1.9831]), 7
    print(f'Predicted: "{predicted}", Actual: "{actual}"')