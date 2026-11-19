"""Unit tests for streaming Ridge logZ calibrator and OOV false-confidence suppression."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from direction1_phase_shortlist.logz_calibrator import (
    calibrated_max_prob_numpy,
    evaluate_logz_calibrator,
    init_logz_accumulator,
    load_logz_calibrator,
    predict_logz_numpy,
    save_logz_calibrator,
    solve_logz_calibrator,
    update_logz_accumulator,
)


def test_streaming_ridge_matches_batch_and_is_immutable() -> None:
    rng = np.random.default_rng(7)
    n_samples, dim = 400, 32
    hidden = rng.normal(size=(n_samples, dim)).astype(np.float32)
    true_w = rng.normal(scale=0.3, size=(dim,))
    true_norm_w = 1.4
    true_bias = 5.2
    targets = (hidden @ true_w) + true_norm_w * np.linalg.norm(hidden, axis=-1) + true_bias

    acc_batch = update_logz_accumulator(init_logz_accumulator(dim), hidden, targets)
    cal_batch = solve_logz_calibrator(acc_batch, ridge_lambda=1e-4)

    acc_stream = init_logz_accumulator(dim)
    for start in range(0, n_samples, 53):
        prev_n = acc_stream.n_samples
        next_acc = update_logz_accumulator(acc_stream, hidden[start : start + 53], targets[start : start + 53])
        assert acc_stream.n_samples == prev_n  # prior accumulator untouched
        acc_stream = next_acc
    cal_stream = solve_logz_calibrator(acc_stream, ridge_lambda=1e-4)

    np.testing.assert_allclose(cal_stream.weight, cal_batch.weight, atol=1e-5)
    assert cal_stream.norm_weight == pytest.approx(cal_batch.norm_weight, abs=1e-5)
    assert cal_stream.bias == pytest.approx(cal_batch.bias, abs=1e-5)

    metrics = evaluate_logz_calibrator(cal_stream, hidden, targets)
    assert metrics.r2 > 0.9999
    assert metrics.mae < 1e-3


def test_radial_norm_feature_captures_symmetric_logsumexp() -> None:
    rng = np.random.default_rng(19)
    dim, vocab = 16, 128
    emb = rng.normal(scale=1.0, size=(vocab, dim))
    # Hidden vectors with varying norms & both signs -> linear-only cannot fit radial growth!
    h_raw = rng.normal(size=(500, dim))
    scales = rng.uniform(0.5, 4.0, size=(500, 1))
    hidden = (h_raw / np.linalg.norm(h_raw, axis=-1, keepdims=True)) * scales

    logits = hidden @ emb.T
    max_l = np.max(logits, axis=-1, keepdims=True)
    true_logz = (max_l[:, 0]) + np.log(np.sum(np.exp(logits - max_l), axis=-1))

    acc = update_logz_accumulator(init_logz_accumulator(dim), hidden[:400], true_logz[:400])
    cal = solve_logz_calibrator(acc, ridge_lambda=1.0)
    held_out = evaluate_logz_calibrator(cal, hidden[400:], true_logz[400:])
    assert cal.norm_weight > 0.5
    assert held_out.r2 > 0.90


def test_calibrated_max_prob_suppresses_oov_false_confidence() -> None:
    # True OOV token has logit 25.0 (logZ_full ~ 25.1), while shortlist only sees max logit 18.0.
    shortlist_logits = np.array([[18.0, 14.0, 13.5, 12.0]], dtype=np.float64)
    _, naive_prob = calibrated_max_prob_numpy(shortlist_logits, predicted_full_logz=-1e9)
    idx, cal_prob = calibrated_max_prob_numpy(shortlist_logits, predicted_full_logz=25.1)

    assert int(idx[0]) == 0
    assert float(naive_prob[0]) > 0.95  # naive softmax dangerously overconfident!
    assert float(cal_prob[0]) < 0.01  # calibrated prob catches the OOV miss immediately!

    # Lower-bound guarantee: even if predicted_full_logz underestimates, prob stays in (0, 1].
    _, bounded_prob = calibrated_max_prob_numpy(shortlist_logits, predicted_full_logz=5.0)
    assert 0.0 < float(bounded_prob[0]) <= 1.0
    assert float(bounded_prob[0]) == pytest.approx(float(naive_prob[0]))

    # Deadband margin protects in-vocab tokens from mild +1.0 predictor noise while still catching huge +7.0 OOV jumps.
    _, guard_in_vocab = calibrated_max_prob_numpy(shortlist_logits, predicted_full_logz=19.0, deadband_margin=2.0)
    assert float(guard_in_vocab[0]) == pytest.approx(float(naive_prob[0]))
    _, guard_oov = calibrated_max_prob_numpy(shortlist_logits, predicted_full_logz=25.1, deadband_margin=2.0)
    assert float(guard_oov[0]) < 0.02

    with pytest.raises(ValueError):
        calibrated_max_prob_numpy(shortlist_logits, predicted_full_logz=20.0, deadband_margin=-0.5)


def test_logz_calibrator_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    hidden = rng.normal(size=(50, 8)).astype(np.float32)
    targets = np.linalg.norm(hidden, axis=-1) + 2.0
    cal = solve_logz_calibrator(update_logz_accumulator(init_logz_accumulator(8), hidden, targets), ridge_lambda=1.0)

    path = save_logz_calibrator(tmp_path / "cal.npz", cal)
    loaded = load_logz_calibrator(path)
    np.testing.assert_allclose(loaded.weight, cal.weight)
    assert loaded.norm_weight == pytest.approx(cal.norm_weight)
    assert loaded.bias == pytest.approx(cal.bias)
    assert loaded.n_samples == cal.n_samples
    assert not loaded.weight.flags.writeable
    np.testing.assert_allclose(predict_logz_numpy(loaded, hidden), predict_logz_numpy(cal, hidden))
