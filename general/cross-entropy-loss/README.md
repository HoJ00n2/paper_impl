## Cross-Entropy Loss

$
\text{Loss} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \cdot \log(\hat{y}_{ij})
$

- $y_{ij}$: i-번째 데이터의 실제 레이블(one-hot encoded)
- $\hat{y}_{ij}$: i-번째 데이터에서 클래스 j의 예측 확률
- N: 데이터 샘플의 수
- C: 클래스의 수

## Code Implementation

### First Code
```python
import numpy as np

def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray):
    N = len(y_true)
    C = len(y_true[0])

    logits = 0
    for i in range(N):
        for j in range(C):
                logits += y_true[i][j] * np.log(y_pred[i][j] + 1e-5)
    
    return (-logits) / N
```

<details>
<summary>Improvments</summary>
<div markdown='1'>

1. **수치 안정성 문제**
    - (y_pred[i][j] + 1e-5): 매우 작은 값을 더해 로그 계산 시 0이 되는 것을 방지하려는 의도는 좋지만, 더 안정적인 방법으로는 np.clip을 사용하는 것이 좋다.
    - np.clip(y_pred, 1e-12, 1 - 1e-12)을 사용하면 입력값이 너무 작거나 너무 커서 계산이 불안정해지는 상황을 방지할 수 있다.
2. **효율성 문제**
    - for 루프를 사용하여 행렬 연산을 반복하면, 데이터 크기가 커질수록 매우 비효율적이다.
    - Numpy의 **벡터화 연산**을 활용하면 for 루프 없이 한 번에 계산할 수 있습니다. 이는 실행 속도를 크게 향상시킨다.
3. **출력 확률 값의 유효성**
    - y_pred가 확률 분포(예: Softmax 출력)인지 확인이 필요합니다. 그렇지 않다면, Softmax를 추가로 적용해야 한다.
    - 만약 y_pred가 Softmax 출력이라면 이 과정을 생략할 수 있다.

</div>
</details>


### Improved Code
```python
import numpy as np

def cross_entroy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Cross-Entropy Loss.

    Parameters:
        y_true (np.ndarray): Ground truth labels in one-hot encoded format with shape [N, C].
                             N is the number of samples, and C is the number of classes.
        y_pred (np.ndarray): Predicted probability distribution from the model (e.g., softmax output)
                             with shape [N, C].

    Returns:
        float: The mean Cross-Entropy Loss across the batch.

    Notes:
        - To ensure numerical stability, the predicted probabilities (y_pred) are clipped to the range [epsilon, 1 - epsilon],
          where epsilon is a small constant (1e-12).
        - The formula for cross-entropy loss is:
          Loss = -1/N * Σ Σ (y_true * log(y_pred))
          where the inner summation is over classes, and the outer summation is over all samples.
    """
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.)
    logits = np.sum(y_true * np.log(y_pred))
    loss = -logits / y_true.shape[0]

    return loss
```