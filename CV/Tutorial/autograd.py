import torch

x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad=True) # weight matrix
b = torch.randn(3, requires_grad=True) # bias
z = torch.matmul(x, w)+ b # 입력 tensor * weight + bias

# loss = torch.nn.functional.binary_cross_entropy(z, y) # input, target은 반드시 0~1사이의 확률값
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) # input, target은 logits 가능!

# 역전파 함수에 대한 ref -> grad_fn 속성으로!
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# gradient 계산하기 dloss/dw, dloss/db
loss.backward()
print(f"dloss/dw : {w.grad}")
print(f"dloss/db : {b.grad}")

# 변화도 추적 멈추기
print(f"with gradient flow : {z.requires_grad}")

with torch.no_grad():
    z = torch.matmul(x, w)+ b
print(f"with no gradient flow : {z.requires_grad}")

# 선택적 읽기 -> 텐서 변화도와 자코비안 곱
inp = torch.eye(4, 5, requires_grad=True) # 4,5의 단위 행렬 생성하고 gradient flow 허용
out = (inp+1).pow(2).t() # (inp+1)^2 후 transpose
print(f"out : {out}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradient\n{inp.grad}")