"""Unit tests for SEED State Sink decomposition and rank-R compression."""

import math
import numpy as np

from seed_agent_lab.seed_ops import compress_seed_state, decompose_state_sink


def test_decompose_state_sink_orthogonality_and_pythagoras() -> None:
    rng = np.random.default_rng(123)
    state = rng.normal(size=(128, 128))
    mean_v = rng.normal(size=(128,))
    dec = decompose_state_sink(state, mean_v)

    # Orthogonality: unit_v_bar^T @ content == 0
    assert np.allclose(dec.unit_v_bar @ dec.content, 0.0, atol=1e-10)
    # Exact reconstruction
    assert np.allclose(dec.reconstruct(), state, atol=1e-10)
    # Energy additivity: ||S||_F^2 == ||a||^2 + ||C||_F^2
    lhs = float(np.sum(state**2))
    rhs = float(np.sum(dec.sink_coeff**2) + np.sum(dec.content**2))
    assert math.isclose(lhs, rhs, rel_tol=1e-9)


def test_compress_seed_state_preserves_sink_and_low_rank_content() -> None:
    rng = np.random.default_rng(7)
    v_bar = rng.normal(size=(64,))
    v_bar /= np.linalg.norm(v_bar)
    a = 50.0 * rng.normal(size=(64,))
    # Content has rank 4 orthogonal to v_bar
    u_raw = rng.normal(size=(64, 4))
    u_orth = u_raw - np.outer(v_bar, v_bar @ u_raw)
    v_content = rng.normal(size=(4, 64))
    state = np.outer(v_bar, a) + u_orth @ v_content

    compressed = compress_seed_state(state, v_bar, rank=4)
    assert np.allclose(compressed, state, atol=1e-10)
