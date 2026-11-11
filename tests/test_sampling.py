import numpy as np

from infer.ops import softmax
from infer.sampling import sample_token


def draw(logits, n, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    return np.array([sample_token(logits, rng, **kwargs) for _ in range(n)])


def test_temperature_sampling_follows_softmax():
    logits = np.array([2.0, 1.0, 0.5, -1.0])
    counts = np.bincount(draw(logits, 20000), minlength=4) / 20000
    np.testing.assert_allclose(counts, softmax(logits), atol=0.015)
    np.testing.assert_array_equal(draw(logits, 50, seed=3), draw(logits, 50, seed=3))
    assert set(draw(logits, 50, temperature=0)) == {0}
    assert (draw(logits, 2000, temperature=0.1) == 0).mean() > 0.99
