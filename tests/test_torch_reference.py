import numpy as np
import torch
import torch.nn.functional as F

from infer.config import ModelConfig
from infer.model import forward, init_weights


def torch_rope(x, positions, base):
    head_dim = x.shape[-1]
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    angles = torch.outer(positions.double(), inv_freq).repeat(1, 2)
    rotated = torch.cat([-x[..., head_dim // 2 :], x[..., : head_dim // 2]], dim=-1)
    return x * angles.cos() + rotated * angles.sin()


def torch_forward(tokens, weights, config):
    w = lambda a: torch.from_numpy(a).double()
    seq = len(tokens)
    positions = torch.arange(seq)
    heads = lambda t, n: t.view(seq, n, config.head_dim).transpose(0, 1)
    norm = lambda t, weight: F.rms_norm(t, (config.dim,), w(weight), eps=config.norm_eps)
    x = w(weights["embed"])[torch.from_numpy(tokens)]
    for layer in weights["layers"]:
        h = norm(x, layer["attn_norm"])
        q = torch_rope(heads(h @ w(layer["wq"]), config.n_heads), positions, config.rope_base)
        k = torch_rope(heads(h @ w(layer["wk"]), config.n_kv_heads), positions, config.rope_base)
        v = heads(h @ w(layer["wv"]), config.n_kv_heads)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + attn.transpose(0, 1).reshape(seq, config.dim) @ w(layer["wo"])
        h = norm(x, layer["mlp_norm"])
        x = x + (F.silu(h @ w(layer["w_gate"])) * (h @ w(layer["w_up"]))) @ w(layer["w_down"])
    return norm(x, weights["final_norm"]) @ w(weights["lm_head"])


def test_numpy_forward_matches_torch_reference():
    config = ModelConfig()
    weights = init_weights(config, seed=7, std=0.2)
    tokens = np.random.default_rng(8).integers(0, config.vocab_size, 12)
    expected = torch_forward(tokens, weights, config).numpy()
    np.testing.assert_allclose(forward(tokens, weights, config), expected, rtol=1e-4, atol=1e-4)
