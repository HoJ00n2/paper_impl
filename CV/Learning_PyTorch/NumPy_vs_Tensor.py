# NumPy로 연산하기
import numpy as np
import math

# 무작위로 입력과 출력 데이터 생성
x = np.linspace(-math.pi, math.pi, 2000)
# np.linspace(하한선, 상한선, list개수) : 하한선에서 상한선까지 차이를 2000등분한 값들 들어감
# 즉, step size는 (상한선 - 하한선)/2000 -> 여기선 (3.14 + 3.14)/2000 씩 증가
# print(len(x), x[:5]) # 2000, -3.14....
y = np.sin(x)
print(f"y : {y}")

# 무작위로 가중치 초기화
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
# 2000 epoch
for t in range(2000):
    # 순전파 단계 : 예측값 y 계산
    # y = a + bx + cx^2 + dx^3
    y_pred = a + b*x + c*x**2 + d*x**3

    # loss 계산
    # y = sin(x)
    loss = np.square(y_pred, y).sum() # ==> (y_pred - y)**2.sum()

    if t % 100 == 99:
        print(t, loss)

    # 손실에 따른 a,b,c,d gradient 계산후 역전파
    grad_y_pred = 2.0 * (y_pred - y) # (y_pred - y)**2 -> y_pred에 대해 미분하면 됨
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 가중치 갱신
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f"Result : y = {a} + {b}x + {c}x^2 + {d}X^3")

# Tensor로 연산하기
import torch

dtype = torch.float
device = torch.device("cpu")

# 무작위 입력과 출력데이터 생성
x_t = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y_t = torch.sin(x_t)

# 무작위로 가중치 초기화
a_t = torch.randn((), device=device, dtype=dtype)
b_t = torch.randn((), device=device, dtype=dtype)
c_t = torch.randn((), device=device, dtype=dtype)
d_t = torch.randn((), device=device, dtype=dtype)

for t in range(2000):
    # 순전파
    y_pred = a_t + b_t*x_t + c_t*x_t**2 + d_t*x_t**3

    # np 안쓸거라 이렇게 구현 (not np.square)
    loss = (y_pred - y_t).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y_t)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x_t).sum()
    grad_c = (grad_y_pred * x_t**2).sum()
    grad_d = (grad_y_pred * x_t**3).sum()

    # 가중치를 갱신
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f"Result : y = {a.item()} + {b.item()}x + {c.item()}x^2 + {d.item()}X^3")
