"""Unit tests for pure functional SVD spectral metrics, state-error decomposition, and weight-sink alignment."""

from __future__ import annotations

import numpy as np
import pytest

from direction3_role_prefill_svd.svd_analysis import (
    analyze_head_weight_sink_alignment,
    compute_singular_spectrum,
    decompose_state_error,
    low_rank_replace_content_error,
)


def test_compute_singular_spectrum_rank1_and_identity() -> None:
    rng = np.random.default_rng(1)
    u = rng.normal(size=(32, 1))
    v = rng.normal(size=(1, 48))
    rank1 = u @ v
    spec1 = compute_singular_spectrum(rank1)
    assert spec1.effective_rank == pytest.approx(1.0, abs=1e-6)
    assert spec1.r50 == 1 and spec1.r90 == 1 and spec1.r99 == 1
    assert spec1.top1_energy == pytest.approx(1.0, abs=1e-9)

    eye16 = np.eye(16, dtype=np.float64)
    spec_id = compute_singular_spectrum(eye16)
    assert spec_id.effective_rank == pytest.approx(16.0, abs=1e-6)
    assert spec_id.r50 == 8
    assert spec_id.r90 == 15
    assert spec_id.top1_energy == pytest.approx(1.0 / 16.0, abs=1e-9)


def test_decompose_state_error_pythagoras_and_low_rank_oracle_recovery() -> None:
    rng = np.random.default_rng(13)
    dv, dk = 32, 32
    mean_v = rng.normal(size=(dv,))
    v_hat = mean_v / np.linalg.norm(mean_v)

    a_true = rng.normal(size=(dk,)) * 5.0
    a_sh = a_true + rng.normal(size=(dk,)) * 0.8

    # Construct orthogonal content matrices with exact rank-4 error E_ax
    p_perp = np.eye(dv) - np.outer(v_hat, v_hat)
    ax_sh = p_perp @ rng.normal(size=(dv, dk))
    e_ax_r4 = p_perp @ (rng.normal(size=(dv, 4)) @ rng.normal(size=(4, dk)))
    ax_true = ax_sh + e_ax_r4

    s_true = np.outer(v_hat, a_true) + ax_true
    s_sh = np.outer(v_hat, a_sh) + ax_sh

    err = decompose_state_error(s_true, s_sh, mean_v)
    assert err.sink_err_norm_sq + err.content_err_norm_sq == pytest.approx(err.total_err_norm_sq, rel=1e-9)
    assert err.sink_err_share + err.content_err_share == pytest.approx(1.0, rel=1e-9)
    assert err.content_err_spectrum.r99 <= 4

    # Exact recovery when rank >= 4 and mean_sink_coeff == a_true
    recovered = low_rank_replace_content_error(s_true, s_sh, mean_v, rank=4, mean_sink_coeff=a_true)
    np.testing.assert_allclose(recovered, s_true.astype(np.float32), atol=1e-4)


def test_analyze_head_weight_sink_alignment_exact_principal_direction() -> None:
    rng = np.random.default_rng(29)
    dv, d_model = 16, 64
    q, _ = np.linalg.qr(rng.normal(size=(dv, dv)))
    v_hat = q[:, 0]

    # Build W_V whose dominant left singular vector is exactly v_hat (sigma_1 = 20.0)
    rhs = rng.normal(size=(dv, d_model))
    rhs, _ = np.linalg.qr(rhs.T)  # (d_model, dv)
    svals = np.array([20.0] + [1.0] * (dv - 1))
    w_v = (q * svals) @ rhs.T

    # Build W_out that suppresses v_hat (near-zero gain along v_hat, unit gain elsewhere)
    w_out = rng.normal(size=(d_model, dv))
    w_out = w_out - np.outer(w_out @ v_hat, v_hat)

    align = analyze_head_weight_sink_alignment(head_idx=3, w_v_head=w_v, w_out_head=w_out, mean_v_head=v_hat * 3.5)
    assert align.head_idx == 3
    assert align.sink_top1_u_cos == pytest.approx(1.0, abs=1e-6)
    assert align.sink_top4_subspace_energy == pytest.approx(1.0, abs=1e-6)
    assert align.sink_wv_energy_fraction == pytest.approx(400.0 / (400.0 + 15.0), rel=1e-6)
    assert align.wout_sink_gain_ratio == pytest.approx(0.0, abs=1e-6)
    # Removing the dominant rank-1 sink direction increases remaining effective rank from ~1.1 to 15.0
    assert align.wv_perp_spectrum.effective_rank == pytest.approx(15.0, abs=1e-4)
