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