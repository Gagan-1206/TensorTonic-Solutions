import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp optimizer update step.

    w : parameters
    g : gradients
    s : running squared gradient accumulator
    """

    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    # Step 1: update running average of squared gradients
    s = beta * s + (1 - beta) * (g ** 2)

    # Step 2: update parameters
    w = w - lr * g / (np.sqrt(s) + eps)

    return w, s