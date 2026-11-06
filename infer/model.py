import numpy as np


def init_weights(config, seed=0, std=0.02):
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    kv_dim = config.n_kv_heads * config.head_dim
    layers = [
        {
            "attn_norm": np.ones(config.dim, dtype=np.float32),
            "wq": normal(config.dim, config.dim),
            "wk": normal(config.dim, kv_dim),
            "wv": normal(config.dim, kv_dim),
            "wo": normal(config.dim, config.dim),
            "mlp_norm": np.ones(config.dim, dtype=np.float32),
            "w_gate": normal(config.dim, config.hidden_dim),
            "w_up": normal(config.dim, config.hidden_dim),
            "w_down": normal(config.hidden_dim, config.dim),
        }
        for _ in range(config.n_layers)
    ]
    return {
        "embed": normal(config.vocab_size, config.dim),
        "layers": layers,
        "final_norm": np.ones(config.dim, dtype=np.float32),
        "lm_head": normal(config.dim, config.vocab_size),
    }
