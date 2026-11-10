import numpy as np

from infer.model import forward


def greedy_decode(prompt, weights, config, max_new_tokens):
    if len(prompt) + max_new_tokens > config.max_positions:
        raise ValueError("prompt plus new tokens exceeds max_positions")
    tokens = list(prompt)
    for _ in range(max_new_tokens):
        logits = forward(np.array(tokens), weights, config)
        tokens.append(int(np.argmax(logits[-1])))
    return tokens[len(prompt) :]
