import numpy as np

from infer.config import ModelConfig
from infer.model import init_weights


def test_init_weights_shapes_and_seed():
    config = ModelConfig()
    weights = init_weights(config, seed=1)
    layer = weights["layers"][0]
    assert len(weights["layers"]) == config.n_layers
    assert weights["embed"].shape == (config.vocab_size, config.dim)
    assert layer["wk"].shape == (config.dim, config.n_kv_heads * config.head_dim)
    assert layer["w_down"].shape == (config.hidden_dim, config.dim)
    assert weights["lm_head"].dtype == np.float32
    np.testing.assert_array_equal(layer["wq"], init_weights(config, seed=1)["layers"][0]["wq"])
