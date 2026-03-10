import numpy as np

def adam_step(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Adam optimizer update step.

    param : current parameters (array)
    grad  : gradients (array)
    m     : first moment vector
    v     : second moment vector
    t     : timestep
    """

    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Parameter update
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param, m, v