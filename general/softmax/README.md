## Softmax

$
\text{softmax}(x_i) = \frac{e^{x_i - \text{max}(x)}}{\sum_j e^{x_j - \text{max}(x)}}
$

## Code Implementation

### First Code
```python
import numpy as np

def softmax(input_array):
    denominator = sum(np.exp(x) for x in input_array)
    numerator = list(map(np.exp, input_array))
    return numerator / denominator
```

<details>
<summary>Improvments</summary>
<div markdown='1'>

---
1. **오버플로 방지**:
    - $e^x$ 는 입력 값이 클수록 오버플로가 발생할 수 있다.
    - 이를 방지하려면 입력 배열에서 최대값을 빼고 지수 연산을 수행해야 한다.
2. **효율성**:
    - map 함수와 리스트를 사용하는 대신, Numpy의 벡터화 연산을 활용하면 속도와 가독성을 동시에 개선할 수 있다.
3. **코드 간결화**:
    - 반복문과 리스트 변환을 제거하고, Numpy의 배열 연산을 활용하여 한 줄로 구현할 수 있다.
---
</div>
</details>

### Improved Code
```python
import numpy as np
from typing import List, Union

def softmax(input_array: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Implementation Softmax

    Parameters:
        input_array (Union[List[float], np.ndarray]): Input vector, list or Numpy array.
    
    Returns:
        np.ndarray: A Numpy array representing a probability distribution.
    """
    input_array = np.array(input_array, dtype=np.float64)
    shifted = input_array - np.max(input_array) # For prevent overflow
    exp_values = np.exp(shifted)
    
    return exp_values / np.sum(exp_values)
```
