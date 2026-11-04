import numpy as np


def softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def silu(x):
    return x / (1 + np.exp(-x))


def swiglu_mlp(x, w_gate, w_up, w_down):
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


def rope_frequencies(head_dim, max_positions, base=10000.0):
    inv_freq = 1.0 / base ** (np.arange(0, head_dim, 2) / head_dim)
    angles = np.outer(np.arange(max_positions), inv_freq)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin, positions):
    c, s = cos[positions], sin[positions]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def causal_mask(q_len, kv_len):
    return np.triu(np.ones((q_len, kv_len), dtype=bool), k=kv_len - q_len + 1)


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    return softmax(scores) @ v


def split_heads(x, n_heads):
    *lead, seq, dim = x.shape
    return np.swapaxes(x.reshape(*lead, seq, n_heads, dim // n_heads), -2, -3)


def merge_heads(x):
    x = np.swapaxes(x, -2, -3)
    return x.reshape(*x.shape[:-2], -1)


def multi_head_attention(x, wq, wk, wv, wo, n_heads, mask=None):
    q, k, v = (split_heads(x @ w, n_heads) for w in (wq, wk, wv))
    return merge_heads(scaled_dot_product_attention(q, k, v, mask)) @ wo
