# autograd의 기본(primitive) 연산자는 실제로 텐서를 조작하는 2개 함수 (forward, backward)
# forward 함수는 입력 텐서로부터 출력 텐서를 계산
# backward 함수는 어떤 스칼라 값에 대한 출력 텐서의 변화도를 받고
# 스칼라 값에 대한 입력 텐서의 변화도 계산 (여기서 스칼라가 뭔데?)
from typing import Any

# PyTorch에선 torch.autograd.Function의 하위클래스(subclass)를 정의하고
# forward, backward를 구현함으로써 사용자 정의 autograd 연산자를 정의 가능
# 그 후 인스턴스를 생성하고 이를 함수처럼 호출하고
# 입력 데이터를 갖는 텐서를 전달하는 식으로 새로운 autograd 연산자 사용 가능

import torch
import math

# torch.autograd.Function의 subclass인 LegendrePolynomial3 class
class LegendrePoylnomial3(torch.autograd.Function):
    """
    torch.autograd.Function을 상속받아 사용자 정의 autograd Function을 구현하고,
    텐서 연산을 하는 순전파 단계와 역전파 단계를 구현
    """

    @staticmethod # 왜 static으로 ?
    def forward(ctx, input):
        """
        순전파 단계에서는 입력을 갖는 텐서를 받아 출력을 갖는 텐서로 반환
        ctx는 컨텍스트 객체 (context object)로 역전파 연산을 위한 정보저장에 사용
        ctx.save_for_backward 메소드를 통해 역전파 단계에 사용할 어떤 객체도
        저장(cache)해 둘 수 있음
        """
        print("Forward called with input:", input) # for debugging
        ctx.save_for_backward(input)
        # 르장드르 다항식인 P3(x)를 의미
        return 0.5 * (5 * input ** 3 - 3 * input)

    # ctx를 통해 저장된 데이터를 읽음 (ctx.saved_tensors)
    # grad_output은 출력(forward의 결과)에 대한 손실의 변화도로 전달됨
    @staticmethod
    def backward(ctx, grad_output):
        """
        역전파 단계에서는 출력에 대한 손실의 변화도를 가지는 텐서를 받고
        입력에 대한 손실의 변화도를 계산
        """
        input, = ctx.saved_tensors # input, 과 input, _ 의 차이는 ?
        print("Backward called with grad_output:", grad_output) # for debugging
        # P3(x)를 x에 대해 편미분한 결과
        return grad_output * 1.5 * (5 * input ** 2 - 1)

dtype = torch.float
device = torch.device("cpu")

# 입력과 출력값을 가지는 텐서 생성
# requires_grad=False가 기본으로 설정되어 역전파 과정에서 변화도 계산 x
# False로 하는 이유는 torch.autograd.Function의 subclass 부르므로 여기서 계산
# -3.14 ~ 3.14 2000간격 만큼의 데이터 생성 (총 2000개 데이터)
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 가중치를 갖는 임의 텐서 생성
# 3차 다항식이므로 4개의 가중치 필요 (0차 = bias, (1,2,3)차 weight) 각 (a,b,c,d)
# 이 가중치들이 수렴하기 위해선 정답으로 부터 너무 멀지 않은 값으로 초기화
# requires_grad = True로 설정하여 역전파 단계중 변화도를 계산
# torch.full(()) 하면 scalar tensor가 나옴 -> 이렇게 한 이유는 torch의 기능을 이용하기 위함 (weight update)
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(1000):
    # 사용자 정의 Function을 적용하기 위해 Function.apply 메소드 사용
    # 여기에 'P3' 이라고 명명
    P3 = LegendrePoylnomial3.apply

    # 순전파 단계 : 연산을 하여 예측값 y 계산
    # 사용자 정의 autograd를 통해 P3 계산
    # P3(c + d * x) 는 P3로 정의된 르장드르 class의 forward에 input으로 (c + d * x)를 넣은 것
    # `apply`를 호출하면 PyTorch가 내부적으로 `forward` 메소드를 호출 -> P3(input)
    y_pred = a + b * P3(c + d * x) # ctx는 PyTorch 내부에서 알아서 할당함

    # 손실을 계산하고 출력
    loss = (y_pred - y).pow(2).sum() # batch내 모든 loss를 합해야 하므로 sum
    if t % 100 == 99:
        print(t, loss.item())

    # autograd를 사용하여 역전파
    # backward 메소드는 PyTorch의 loss.backward() 호출 시 자동으로 실행됩니다.
    # loss.backward()는 computational graph를 통해 gradient를 (ex : a.grad) 구하는 과정만 함 (업데이트를 하지는 않음)
    loss.backward()

    # 경사하강법을 통해 가중치 갱신 -> 여긴 왜 no_grad?
    # loss.backward()를 통해 구한 grad를 여기서 weight update (optimizer step 과정이 여기에 해당)
    with torch.no_grad():
        # "learning rate * weight's gradient" 값으로 업데이트
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 가중치 갱신 이후에는 초기화
        a.grad, b.grad, c.grad, d.grad = None, None, None, None

print(f"Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()}x)")