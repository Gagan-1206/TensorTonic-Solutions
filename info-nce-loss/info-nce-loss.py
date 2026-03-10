import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE loss for contrastive learning.
    
    Z1: (N, D) embedding batch
    Z2: (N, D) embedding batch
    temperature: float
    """

    Z1 = np.array(Z1, dtype=float)
    Z2 = np.array(Z2, dtype=float)

    # Similarity matrix
    S = np.dot(Z1, Z2.T) / temperature   # (N, N)

    # Numerical stability
    S = S - np.max(S, axis=1, keepdims=True)

    # Exponentiate
    exp_S = np.exp(S)

    # Softmax denominator
    denom = np.sum(exp_S, axis=1)

    # Positive pair scores (diagonal)
    pos = np.diag(exp_S)

    # InfoNCE loss
    loss = -np.log(pos / denom)

    return float(np.mean(loss))