import torch
import math

class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        생성자에서 4개의 매개변수를 생성(instantiate)하고, 멤버 변수로 지정
        """
        super().__init__() # torch.nn.Module 기능들 상속받기
        # weight parameter 설정
        # x는 입력 데이터의 tensor
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        순전파 함수에서는 입력 데이터의 텐서를 받고 출력 데이터의 텐서를 반환
        텐서들 간의 임의의 연산 뿐만 아니라, 생성자에서 정의한 Module을 사용 가능
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        Python의 다른 클래스처럼, PyTorch 모듈을 사용해서 사용자 정의 메소드 정의 가능
        """
        return f'y = {self.a.item()} + {self.b.item()}x + {self.c.item()}x^2 + {self.d.item()}x^3'

# 입력값과 출력값을 갖는 텐서를 생성
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 위에서 정의한 클래스로 모델 생성
model = Polynomial3()

# 손실 함수와 optimizer 생성
# SGD 생성자에 model.parameters()를 호출하면
# 모델의 학습 가능한(=torch.nn.Parameter로 정의된) 매개변수들이 포함
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(1000):
    # 순전파 단계: 모델에 x를 전달하여 예측값 y를 계산합니다.
    y_pred = model(x)

    # 손실을 계산하고 출력
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 변화도를 0으로 만들고, 역전파 수행하고 ,가중치 갱신
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Result: {model.string()}")
