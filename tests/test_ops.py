import numpy as np
import torch

from infer.ops import rms_norm, softmax


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
