import numpy as np

from infer.ops import softmax


def top_k_filter(logits, k):
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -k)[-k]
    return np.where(logits < kth, -np.inf, logits)


def sample_token(logits, rng, temperature=1.0, top_k=0):
    logits = top_k_filter(logits, top_k)
    if temperature == 0:
        return int(np.argmax(logits))
    probs = softmax(logits.astype(np.float64) / temperature)
    return int(rng.choice(len(probs), p=probs))
