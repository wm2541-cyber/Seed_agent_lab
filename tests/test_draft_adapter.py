"""Unit tests for streaming Ridge accumulator and MLX Draft Adapters."""

import mlx.core as mx
import numpy as np

from seed_agent_lab.draft_adapter import (
    LinearDraftAdapter,
    SwiGLUDraftAdapter,
    init_ridge_accumulator,
    solve_ridge_accumulator,
    update_ridge_accumulator,
)
from seed_agent_lab.rotation import apply_linear_rotation, fit_linear_rotation


def test_streaming_ridge_matches_batch_ridge() -> None:
    rng = np.random.default_rng(11)
    src = rng.normal(size=(120, 16))
    tgt = rng.normal(size=(120, 16))

    batch_map = fit_linear_rotation(src, tgt, ridge_lambda=25.0)

    acc = init_ridge_accumulator(16)
    acc = update_ridge_accumulator(acc, src[:50], tgt[:50])
    acc = update_ridge_accumulator(acc, src[50:], tgt[50:])
    stream_map = solve_ridge_accumulator(acc, ridge_lambda=25.0)

    assert np.allclose(stream_map.weight_with_bias, batch_map.weight_with_bias, atol=1e-10)


def test_adapters_zero_init_matches_ridge() -> None:
    rng = np.random.default_rng(3)
    src = rng.normal(size=(40, 16))
    tgt = rng.normal(size=(40, 16))
    rot = fit_linear_rotation(src, tgt, ridge_lambda=10.0)

    lin = LinearDraftAdapter(16, init_map=rot)
    swiglu = SwiGLUDraftAdapter(16, bottleneck_dim=8, init_map=rot)
    x_mx = mx.array(src[:5], dtype=mx.float32)

    expected = apply_linear_rotation(rot, src[:5])
    out_lin = np.array(lin(x_mx))
    out_swi = np.array(swiglu(x_mx))

    assert np.allclose(out_lin, expected, atol=1e-5)
    assert np.allclose(out_swi, expected, atol=1e-5)
