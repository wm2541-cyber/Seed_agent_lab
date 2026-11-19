"""Unit tests for closed-form linear rotation probes."""

import numpy as np

from seed_agent_lab.rotation import apply_linear_rotation, fit_linear_rotation


def test_linear_rotation_recovers_orthogonal_transform() -> None:
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    bias = rng.normal(size=(1, 16))
    src = rng.normal(size=(200, 16))
    tgt = src @ q + bias

    rot = fit_linear_rotation(src[:150], tgt[:150], ridge_lambda=1e-4)
    pred = apply_linear_rotation(rot, src[150:])
    rel_err = np.linalg.norm(pred - tgt[150:]) / np.linalg.norm(tgt[150:])
    assert rel_err < 1e-4
