import torch
import numpy as np

# initializing a Tensor
# 다양한 방법으로 텐서 초기화 가능
# list -> tensor
data = [[1, 2], [3, 4]] # tensor.shpae : [2,2]
x_data = torch.tensor(data)

# Numpy to tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# another tensor to tensor
# 새로운 텐서는 명시적으로 재정의하지 않는 한,
# 인자로 전달된 텐서의 속성(형태와 데이터 타입)을 그대로 유지합니다.
x_ones = torch.ones_like(x_data) # x_data의 속성 그대로 유지
print(f"One Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 datatype 오버라이드
print(f"Random Tensor: \n {x_rand} \n")

# 난수나 상수로 설정
# shape는 tensor의 차원을 튜플로 보여줌
shape = (2,3,)
rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones_like(shape) # ones_like() 인자는 tensor만 받음, tuple은 불가
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# tensor에서 가장 바깥을 감싸는 []는 무시하고 생각
# 그렇게하면 숫자를 감싼 []는 2개가 있고, 각 []안은 3개의 원소가 있으므로 [2,3]
print(f"rand_tensor : {rand_tensor}") # tensor([[0.4312, 0.4522, 0.7323],
                                      # [0.6891, 0.8696, 0.4610]])
print(f"rand_tensor.shape : {rand_tensor.shape}") # torch.size([2,3])

# Tensor의 Attributes
# Tensor의 attributes는 shape, datatype, 저장된 device를 알려줌
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Tensor 연산
# 이 연산들은 모두 GPU에서도 실행할 수 있으며, 대개 CPU보다 더 빠르게 수행됩니다.
# 기본적으로, 텐서는 CPU에서 생성됩니다. GPU를 사용하려면
# .to 메서드를 사용해 텐서를 명시적으로 GPU로 이동시켜야 합니다.
# 다만, 큰 텐서를 디바이스 간에 복사하는 작업은 시간과 메모리 측면에서 비용이 많이 들 수 있다는 점을 염두에 두세요!
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# standard numpy-like indexing and slicing:
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}") # ":"은 모든 차원을 선택하겠단 의미
print(f"Last column: {tensor[..., -1]}") # "..."은 모든 차원을 의미, ":"와 같음
tensor[:,1] = 0 # 모든 행의 1번째 행만 0으로 바꿈
print(tensor) # 의도대로 나옴

# tensor 결합 (torch.cat, torch.stack)
# tensor.shape : ([4,4])
t1 = torch.cat([tensor, tensor, tensor], dim=1) # 열방향으로 늘리기
print(t1.shape) # [4, 12]
t2 = torch.cat([tensor, tensor, tensor]) # 행방향으로 tensor 쌓기
print(t2.shape) # [12, 4]

t3 = torch.stack([tensor, tensor, tensor]) # 0번째 차원 방향으로 [4,4] tensor stack하기
print(t3.shape) # [3,4,4]
t4 = torch.stack([tensor, tensor, tensor], dim=1) # 1번째 [4,4] tensor stack
print(t4.shape) # [4,3,4] >> 어떻게 생긴건지 감이 안오네

# 텐서 산술연산
# 두 텐서사이의 행렬곱을 표현하는 다양한 방식 -> y1,y2,y3는 같은 결과를 보임
# tensor.T 는 tensor의 transpose를 return
y1 = tensor @ tensor.T # 1. @를 활용
y2 = tensor.matmul(tensor.T) # tensor1.matmul(tensor2) 활용

y3 = torch.rand_like(y1) # y1의 shape를 하며 각 random값을 가지는 y3
torch.matmul(tensor, tensor.T, out=y3) # torch.matmul(tensor1, tensor2) 활용

# Single-element tensors
# 1개의 value만을 가지는 tensor가 있다면, ex) loss
# 이를 item() 메서드를 사용하여 Python의 숫자 값으로 변환 가능 (tensor와 numeric은 엄연히 다름)
agg = tensor.sum()
agg_item = agg.item()
print(type(agg), type(agg_item)) #  <torch.tensor>, <float>

# In-place operations
# 결과를 피연산자(operand)에 직접 저장하는 연산을 제자리 연산(in-place operation)
# 이러한 연산은 이름에 _ 접미사를 가지고 있습니다.
# 예를 들어: x.copy_(y), x.t_()는 x의 값을 변경합니다.
tensor.add(5)
print(f"not inplace operation result : {tensor}") # 원본 변경 x
tensor.add_(5)
print(f"inplace operation result : {tensor}") # 원본 변경 o

# Bridge with NumPy
# CPU상의 tensor나 NumPy array는 같은 메모리 공간을 공유 가능
# 그러므로 값이 바뀌면 서로 영향을 줌
# Tensor to NumPy array
t = torch.ones(5)
print(f"t : {t}")
n = t.numpy()
print(f"n : {n}")

# 같은 메모리를 공유하는지 테스트 -> 공유!
t.add_(1)
print(f"t : {t}")
print(f"n : {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

# 같은 메모리 공유하는지 테스트 -> 공유!
np.add(n, 1, out=n)
print(f"t : {t}")
print(f"n : {n}")