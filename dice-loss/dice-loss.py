import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    Works for 1D or 2D inputs.
    """

    # Convert to float NumPy arrays
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Flatten to handle any shape uniformly
    p = p.ravel()
    y = y.ravel()

    # Intersection and sums
    intersection = np.sum(p * y)
    total = np.sum(p) + np.sum(y)

    # Dice coefficient
    dice = (2.0 * intersection + eps) / (total + eps)

    # Dice loss
    return 1.0 - dice