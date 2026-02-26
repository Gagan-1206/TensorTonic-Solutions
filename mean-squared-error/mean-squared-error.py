import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """

    # Convert to NumPy arrays
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    # Shape check
    if y_pred.shape != y_true.shape:
        return None

    # Compute MSE
    mse = np.mean((y_pred - y_true) ** 2)

    return float(mse)
