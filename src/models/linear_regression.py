import numpy as np

def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Compute predictions for every row in X, given weights w and bias b.
    """
    predictions = X @ w + b
    return predictions

def compute_gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """
    Compute the gradient of MSE loss with respect to w and b, given the
    current weights/bias and a batch of training data.
    """
    n = X.shape[0]
    predictions = predict(X, w, b)
    errors = predictions - y

    grad_w = (2 / n) * (X.T @ errors)
    grad_b = (2 / n) * errors.sum()

    return grad_w, grad_b

def compute_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Compute Mean Squared Error between predictions and actual values.
    """
    predictions = predict(X, w, b)
    return np.mean((predictions - y) ** 2)

def fit(X: np.ndarray, y: np.ndarray, learning_rate: float, num_iterations: int) -> tuple[np.ndarray, float, list[float]]:
    """
    Train linear regression weights via gradient descent. Returns the
    final w, b, and the loss recorded at every iteration (for diagnosing
    whether training actually converged).
    """
    w = np.zeros(X.shape[1])
    b = 0.0
    loss_history = []
    for i in range(0, num_iterations):
        grad_w, grad_b = compute_gradients(X, y, w, b)
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        loss_history.append(compute_loss(X, y, w, b))
    return w, b, loss_history
