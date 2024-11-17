import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

device = torch.device("cpu")

train_loaders = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50) # 320 -> 500
        self.fc2 = nn.Linear(50, 10)

        self.localization = nn.Sequential(
            nn.Conv2d(3,8, kernel_size=6),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=4),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # theta 값 regression(예측)
        self.fc_loc = nn.Sequential(
            # nn.Linear(10*3*3, 32), # for MNIST
            nn.Linear(10*5*5, 32), #
            nn.ReLU(True),
            nn.Linear(32, 3*2) # affine이 3x2 행렬이므로 이렇게
        )

        # identity transformation
        print(f"fc_loc : {self.fc_loc}")
        print(f"fc_loc[2] : {self.fc_loc[2]}")
        self.fc_loc[2].weight.data.zero_()
        # (2)번 수식에 1번째 요소와 5번째 요소가 1이어야 행렬곱할 때 x,y 부분만 적용됨 => 항등 변환을 위함
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))

    # STN의 forward 함수
    def stn(self, x):
        xs = self.localization(x)
        print(f"xs.shape : {xs.shape}")
        # for MNIST
        # xs = xs.view(-1, 10 * 3 * 3) # view로 10*3*3로 해주는 이유는 fc_loc에 넣을 때 차원 맞추기 위함

        # for CIFAR10
        xs = xs.view(-1, 10 * 5 * 5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3) # 2행 3렬의 행렬로 변환 -> (affine) == A_theta 행렬

        # affine_grid : 애초에 입력값인 theta가 2D에서는 (N x 2 x 3) 고정
        grid = F.affine_grid(theta, x.size()) # affine 변환 적용

        # grid_sample : 보간법을 적용하기 위함 (defulat가 nearest neighnor 대신 bilinear 인 듯 ?)
        # input feature map x를 affine 변환된 grid로 매핑할 때, 매핑이 잘 안되는 픽셀에대한 (소수점 등..) 보간법을 적용 by grid_sample
        x = F.grid_sample(x, grid) # 그리드 샘플링 (bounding box만큼만 추출)
        return x # 보간법이 적용된 bounding box 영역의 모든 point return

    def forward(self, x):
        x = self.stn(x)

        # 여기까진 일반적인 CNN 흐름
        x = F.relu(F.max_pool2d(self.conv1(x),2)) # conv -> maxpooling -> relu 한 세트에 적용
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2)) # conv -> dropout -> maxpooling -> relu 적용

        # for MNIST
        # x = x.view(-1,320) # 위 결과로 인한 batch제외 h,w,c의 곱은 320이 나옴 -> fc layer로 연산 해주기 위함
        # for CIFAR10
        x = x.view(-1, 500)

        x = F.relu(self.fc1(x)) # 320 -> 50
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) # 50 -> 10 (최종 10개)

        # 근데 여기서 log_softmax로 반환하는 이유 ?
        # 이후 loss에서 nll_loss를 사용할 것이기 때문! 이때는 log_softmax를 해주는 과정이 필수적 (미리 log로 만들기)
        # 10개 클래스에 대한 각각의 log_softmax(추론 값) 적용
        return F.log_softmax(x, dim=1)

model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loaders):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data) # model에 데이터 넣고 나온 추론값
        loss = F.nll_loss(output, target) # 예측값 vs 정답 loss 구하기
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({: .0f})%]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders.dataset),
                100. * batch_idx / len(train_loaders), loss.item()
            ))

def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 배치 손실 합하기
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # 로그 -확률의 최대값에 해당하는 인덱스 가져오기
            print(f"output.max : {output.max(1, keepdim=True)}")
            # output.max (batch중 log 확률 최대값, 해당 최대값의 idx)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}. Accuracy: {}/{} ({:.0f}%\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# 학습 후 공간 변환 계층의 출력을 시각화하고, 입력 이미지 배치 데이터 및
# STN을 사용해 변환된 배치 데이터를 시각화 합니다.


def visualize_stn():
    with torch.no_grad():
        # 학습 데이터의 배치 가져오기
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # 결과를 나란히 표시하기
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

# 학습
for epoch in range(1, 20 + 1):
    train(epoch)
    test()

visualize_stn()

plt.ioff()
plt.show()