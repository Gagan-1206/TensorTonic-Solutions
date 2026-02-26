import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    # Convert to NumPy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)

    # Ensure a and b are 2D (N, D)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]

    # Broadcast if needed
    diff = a - b

    # Euclidean distance for each pair
    d = np.linalg.norm(diff, axis=1)

    # Validate labels
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")

    # Positive pairs → y * d^2
    pos_loss = y * (d ** 2)

    # Negative pairs → (1 - y) * max(0, margin - d)^2
    neg_loss = (1 - y) * np.maximum(0, margin - d) ** 2

    loss = pos_loss + neg_loss

    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")