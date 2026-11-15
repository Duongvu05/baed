"""The vectorised macro-F1 threshold search must match the per-threshold sklearn loop."""
import numpy as np
from sklearn.metrics import f1_score

from baed.pipelines import detect as D


def _reference(y, prob):
    cand = D._threshold_candidates(prob)
    f1s = [f1_score(y, prob > t, average="macro") for t in cand]
    return float(cand[int(np.argmax(f1s))]), np.array(f1s)


def test_same_threshold_and_curve():
    rng = np.random.default_rng(0)
    for trial in range(40):
        n = int(rng.integers(5, 400))
        y = (rng.random(n) < rng.uniform(0.02, 0.5)).astype(int)
        if y.min() == y.max():
            y[0] = 1 - y[0]
        prob = rng.random(n)
        if trial % 3 == 0:
            prob = np.round(prob, 1)  # many ties
        if trial % 5 == 0:
            prob = np.clip(prob + 0.5 * y, 0, 1)  # informative scores
        ref_t, ref_curve = _reference(y, prob)
        curve = D.macro_f1_curve(y, prob, D._threshold_candidates(prob))
        assert np.allclose(curve, ref_curve, atol=1e-12)
        assert D.val_threshold(y, prob) == ref_t
