import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Write code here
    import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein critic loss.
    L = mean(fake_scores) - mean(real_scores)
    """

    real_scores = np.array(real_scores, dtype=float)
    fake_scores = np.array(fake_scores, dtype=float)

    loss = np.mean(fake_scores) - np.mean(real_scores)

    return float(loss)