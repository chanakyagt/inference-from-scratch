import pytest

from infer.config import ModelConfig


def test_model_config_head_dim_and_validation():
    assert ModelConfig().head_dim == 16
    with pytest.raises(ValueError):
        ModelConfig(n_heads=4, n_kv_heads=3)
