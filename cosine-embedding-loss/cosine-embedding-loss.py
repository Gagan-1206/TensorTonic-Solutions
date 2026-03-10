import math

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """

    # dot product
    dot = sum(a*b for a, b in zip(x1, x2))

    # vector norms
    norm1 = math.sqrt(sum(a*a for a in x1))
    norm2 = math.sqrt(sum(b*b for b in x2))

    # cosine similarity
    cos_sim = dot / (norm1 * norm2)

    # compute loss
    if label == 1:
        return float(1 - cos_sim)
    else:
        return float(max(0.0, cos_sim - margin))