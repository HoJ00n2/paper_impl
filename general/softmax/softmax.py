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