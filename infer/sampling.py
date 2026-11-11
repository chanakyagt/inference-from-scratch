import numpy as np

from infer.ops import softmax


def sample_token(logits, rng, temperature=1.0):
    if temperature == 0:
        return int(np.argmax(logits))
    probs = softmax(logits.astype(np.float64) / temperature)
    return int(rng.choice(len(probs), p=probs))
