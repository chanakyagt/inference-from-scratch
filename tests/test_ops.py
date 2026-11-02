import numpy as np
import torch

from infer.ops import rms_norm, silu, softmax, swiglu_mlp


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
