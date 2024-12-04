import numpy as np
import math

# 무작위로 입력과 출력 데이터를 생성
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# 무작위로 가중치를 초기화
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # 예측값 y 계산
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    loss = (y_pred-y).sum()
    if t % 100 == 99:
        print(t, loss)

    # 각각 이렇게 일일이 계산한 이유는 a,b,c,d가 torch형이 아닌 numpy이기 때문
    # numpy에서는 연산 그래프나, gradient, 딥러닝 기능을 제공하지 않음!

    # loss에 따른 weight grad 계산 == loss.backward()
    # 각 수식은 weight로 미분한 편미분 결과
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 가중치 갱신 == optimizer.step()
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
