import numpy as np

from infer.config import ModelConfig
from infer.model import init_weights, transformer_block
from infer.ops import rope_frequencies


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


def test_transformer_block_residuals_and_causality():
    config = ModelConfig()
    layer = init_weights(config, seed=2)["layers"][0]
    cos, sin = rope_frequencies(config.head_dim, config.max_positions, config.rope_base)
    x = np.random.default_rng(3).standard_normal((6, config.dim)).astype(np.float32)
    out = transformer_block(x, layer, config, cos, sin, np.arange(6))
    assert out.shape == x.shape
    np.testing.assert_allclose(transformer_block(x[:4], layer, config, cos, sin, np.arange(4)), out[:4], rtol=1e-5, atol=1e-6)
    silent = dict(layer, wo=np.zeros_like(layer["wo"]), w_down=np.zeros_like(layer["w_down"]))
    np.testing.assert_array_equal(transformer_block(x, silent, config, cos, sin, np.arange(6)), x)
