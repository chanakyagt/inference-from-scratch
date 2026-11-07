import numpy as np

from infer.ops import apply_rope, causal_mask, merge_heads, repeat_kv, rms_norm, scaled_dot_product_attention, split_heads, swiglu_mlp


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


def rope_attention(x, layer, config, cos, sin, positions):
    q = apply_rope(split_heads(x @ layer["wq"], config.n_heads), cos, sin, positions)
    k = apply_rope(split_heads(x @ layer["wk"], config.n_kv_heads), cos, sin, positions)
    v = split_heads(x @ layer["wv"], config.n_kv_heads)
    n_rep = config.n_heads // config.n_kv_heads
    mask = causal_mask(x.shape[-2], x.shape[-2])
    out = scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), mask)
    return merge_heads(out) @ layer["wo"]


def transformer_block(x, layer, config, cos, sin, positions):
    h = x + rope_attention(rms_norm(x, layer["attn_norm"], config.norm_eps), layer, config, cos, sin, positions)
    mlp_in = rms_norm(h, layer["mlp_norm"], config.norm_eps)
    return h + swiglu_mlp(mlp_in, layer["w_gate"], layer["w_up"], layer["w_down"])
