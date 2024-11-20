## Gradient Descent
$w := w - \eta \cdot \frac{\partial L}{\partial w}$

- $w$ : 모델의 파라미터 (가중치)
- $\eta$ : 학습률 (Learning Rate)
- $\frac{\partial L}{\partial w}$ : 손실 함수 $L$에 대한 $w$의 기울기

## Code Implementation
```python
import torch

# sample data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]], requires_grad=False)

# model init : y = ax + b
weight = torch.randn(1, 1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)

# Loss : MSE
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Gradient Descent Implementation
learning_rate = 0.01

for epoch in range(1000):
    y_pred = x.mm(weight) + bias    # wx + b
    loss = mse_loss(y_pred, y)
    loss.backward() # backpropagation

    # Gradient Descent
    with torch.no_grad():
        weight -= learning_rate * weight.grad # update weight
        bias -= learning_rate * bias.grad # update bias
    
    # Init gradient
    weight.grad.zero_()
    bias.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print(f"Learned Weight: {weight.item()}")
print(f"Learned Bias: {bias.item()}")

# Test
test_x = torch.tensor(([[5.0]]))
test_y_pred = test_x.mm(weight) + bias
print(f"Prediction for input 5.0: {test_y_pred}")
```
```
>>>
Epoch 0, Loss: 38.80702590942383
Epoch 100, Loss: 0.03523928299546242
Epoch 200, Loss: 0.019345970824360847
Epoch 300, Loss: 0.010620692744851112
Epoch 400, Loss: 0.005830593407154083
Epoch 500, Loss: 0.0032009370625019073
Epoch 600, Loss: 0.001757259014993906
Epoch 700, Loss: 0.0009647055412642658
Epoch 800, Loss: 0.0005296136368997395
Epoch 900, Loss: 0.0002907567541114986
Learned Weight: 1.989485263824463
Learned Bias: 1.0309149026870728
Prediction for input 5.0: tensor([[10.9783]], grad_fn=<AddBackward0>)
```