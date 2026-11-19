"""Unit tests for shared-Gram streaming Ridge, SVD low-rank factorization, and sink-orthogonal projection."""

from __future__ import annotations

import numpy as np
import pytest

from direction2_hybrid_prefill_align.streaming_ridge import (
    apply_affine_numpy,
    apply_low_rank_numpy,
    factorize_affine_svd,
    init_shared_gram,
    project_out_head_sinks,
    solve_shared_gram_ridge,
    update_shared_gram,
)


def test_shared_gram_streaming_matches_batch_and_is_immutable() -> None:
    rng = np.random.default_rng(42)
    n, d_in, d_k, d_v = 300, 24, 12, 16
    x = rng.normal(size=(n, d_in)).astype(np.float32)
    w_k = rng.normal(scale=0.4, size=(d_in, d_k)).astype(np.float32)
    b_k = rng.normal(scale=1.0, size=(d_k,)).astype(np.float32)
    w_v = rng.normal(scale=0.3, size=(d_in, d_v)).astype(np.float32)
    b_v = rng.normal(scale=0.5, size=(d_v,)).astype(np.float32)
    yk = (x @ w_k) + b_k
    yv = (x @ w_v) + b_v

    acc_batch = update_shared_gram(init_shared_gram(d_in, {"k": d_k, "v": d_v}), x, {"k": yk, "v": yv})
    sol_batch = solve_shared_gram_ridge(acc_batch, ridge_lambda=1e-3)

    acc_stream = init_shared_gram(d_in, {"k": d_k, "v": d_v})
    for start in range(0, n, 37):
        prev_n = acc_stream.n_samples
        next_acc = update_shared_gram(
            acc_stream,
            x[start : start + 37],
            {"k": yk[start : start + 37], "v": yv[start : start + 37]},
        )
        assert acc_stream.n_samples == prev_n  # prior accumulator untouched
        acc_stream = next_acc
    sol_stream = solve_shared_gram_ridge(acc_stream, ridge_lambda=1e-3)

    np.testing.assert_allclose(sol_stream["k"].weight, sol_batch["k"].weight, atol=1e-4)
    np.testing.assert_allclose(sol_stream["k"].bias, sol_batch["k"].bias, atol=1e-4)
    np.testing.assert_allclose(sol_stream["v"].weight, sol_batch["v"].weight, atol=1e-4)
    np.testing.assert_allclose(apply_affine_numpy(sol_stream["k"], x), yk, atol=1e-2)
    assert not sol_stream["k"].weight.flags.writeable


def test_factorize_affine_svd_and_sink_projection() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 20)).astype(np.float32)
    # Exact rank-4 linear target
    w_true = (rng.normal(size=(20, 4)) @ rng.normal(size=(4, 16))).astype(np.float32)
    b_true = rng.normal(size=(16,)).astype(np.float32)
    y = (x @ w_true) + b_true

    acc = update_shared_gram(init_shared_gram(20, {"y": 16}), x, {"y": y})
    aff = solve_shared_gram_ridge(acc, ridge_lambda=1e-5)["y"]

    lr2 = factorize_affine_svd(aff, rank=2)
    lr4 = factorize_affine_svd(aff, rank=4)
    assert lr2.retained_energy < lr4.retained_energy
    assert lr4.retained_energy == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(apply_low_rank_numpy(lr4, x), apply_affine_numpy(aff, x), atol=1e-4)

    # Sink orthogonal projection: exact zero dot with each head's v_hat and idempotent
    hv, dv = 4, 8
    mean_v = rng.normal(size=(hv, dv))
    v_hat = mean_v / np.linalg.norm(mean_v, axis=-1, keepdims=True)
    vecs = rng.normal(size=(10, hv * dv)).astype(np.float32)
    perp1 = project_out_head_sinks(vecs, mean_v)
    perp2 = project_out_head_sinks(perp1, mean_v)
    np.testing.assert_allclose(perp1, perp2, atol=1e-5)
    dots = np.sum(perp1.reshape(10, hv, dv) * v_hat[None, :, :], axis=-1)
    assert float(np.max(np.abs(dots))) < 1e-5
