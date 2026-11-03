import numpy as np
import torch

from infer.ops import apply_rope, rms_norm, rope_frequencies, silu, softmax, swiglu_mlp


def test_softmax_matches_torch():
    x = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32) * 30
    expected = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
    np.testing.assert_allclose(softmax(x), expected, rtol=1e-5, atol=1e-6)


def test_rms_norm_matches_torch():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 8)).astype(np.float32)
    w = rng.standard_normal(8).astype(np.float32)
    expected = torch.nn.functional.rms_norm(torch.from_numpy(x), (8,), torch.from_numpy(w), eps=1e-6).numpy()
    np.testing.assert_allclose(rms_norm(x, w), expected, rtol=1e-5, atol=1e-6)


def test_swiglu_mlp_matches_torch():
    rng = np.random.default_rng(2)
    x, w_gate, w_up, w_down = (rng.standard_normal(s).astype(np.float32) for s in [(5, 8), (8, 16), (8, 16), (16, 8)])
    t = lambda a: torch.from_numpy(a)
    expected = ((torch.nn.functional.silu(t(x) @ t(w_gate)) * (t(x) @ t(w_up))) @ t(w_down)).numpy()
    np.testing.assert_allclose(silu(x), torch.nn.functional.silu(t(x)).numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(swiglu_mlp(x, w_gate, w_up, w_down), expected, rtol=1e-4, atol=1e-4)


def test_rope_matches_complex_rotation():
    x = np.random.default_rng(3).standard_normal((2, 6, 16))
    cos, sin = rope_frequencies(16, 32)
    positions = np.array([0, 3, 5, 9, 20, 31])
    t = torch.from_numpy(x)
    angles = torch.from_numpy(np.arctan2(sin, cos)[positions])
    rotated = torch.complex(t[..., :8], t[..., 8:]) * torch.polar(torch.ones_like(angles), angles)
    expected = torch.cat([rotated.real, rotated.imag], dim=-1).numpy()
    np.testing.assert_allclose(apply_rope(x, cos, sin, positions), expected, rtol=1e-10, atol=1e-10)


def test_rope_scores_depend_only_on_relative_position():
    q, k = np.random.default_rng(4).standard_normal((2, 1, 16))
    cos, sin = rope_frequencies(16, 64)
    score = lambda m, n: (apply_rope(q, cos, sin, [m]) @ apply_rope(k, cos, sin, [n]).T).item()
    np.testing.assert_allclose(np.linalg.norm(apply_rope(q, cos, sin, [7])), np.linalg.norm(q))
    np.testing.assert_allclose(score(10, 3), score(40, 33))
