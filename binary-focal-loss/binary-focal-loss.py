import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """

    # convert to numpy arrays
    predictions = np.array(predictions, dtype=float)
    targets = np.array(targets, dtype=float)

    # probability of the true class
    p_t = np.where(targets == 1, predictions, 1 - predictions)

    # focal loss formula
    loss = -alpha * (1 - p_t) ** gamma * np.log(p_t)

    # return mean loss
    return float(np.mean(loss))