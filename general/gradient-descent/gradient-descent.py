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