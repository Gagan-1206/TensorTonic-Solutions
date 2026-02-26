import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    p : array-like, shape (N,) -> predicted probabilities (0 < p < 1)
    y : array-like, shape (N,) -> binary labels {0,1}
    gamma : focusing parameter
    """

    # Convert to NumPy arrays
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Elementwise focal loss
    loss = (
        - (1 - p) ** gamma * y * np.log(p)
        - (p ** gamma) * (1 - y) * np.log(1 - p)
    )

    return np.mean(loss)