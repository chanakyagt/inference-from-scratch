import numpy as np

from infer.ops import softmax
from infer.sampling import sample_token, top_k_filter


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


def test_top_k_only_samples_the_k_best():
    logits = np.array([0.3, 2.0, -1.0, 1.5, 0.9])
    assert np.isfinite(top_k_filter(logits, 2)).sum() == 2
    assert set(draw(logits, 3000, top_k=2)) == {1, 3}
    assert set(draw(logits, 200, top_k=1)) == {1}
    np.testing.assert_array_equal(top_k_filter(logits, 0), logits)
