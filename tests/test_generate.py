import numpy as np
import pytest

from infer.config import ModelConfig
from infer.generate import greedy_decode
from infer.model import forward, init_weights


def test_greedy_decode_picks_argmax_each_step():
    config = ModelConfig()
    weights = init_weights(config, seed=5, std=0.2)
    prompt = [1, 2, 3]
    out = greedy_decode(prompt, weights, config, 6)
    assert len(out) == 6
    for i, token in enumerate(out):
        assert token == int(np.argmax(forward(np.array(prompt + out[:i]), weights, config)[-1]))
    with pytest.raises(ValueError):
        greedy_decode(prompt, weights, config, config.max_positions)
