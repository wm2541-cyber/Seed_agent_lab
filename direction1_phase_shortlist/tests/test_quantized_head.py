"""Unit tests for bit-exact MLX QuantizedEmbedding row slicing and logZ-calibrated draft prediction."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from direction1_phase_shortlist import ensure_metal_gpu_arch
from direction1_phase_shortlist.logz_calibrator import LogZCalibrator
from direction1_phase_shortlist.quantized_head import (
    build_prompt_sub_head,
    compile_mlx_logz_head,
    predict_full_head_draft,
    predict_shortlist_draft,
    project_shortlist_logits,
    slice_quantized_embedding,
)
from direction1_phase_shortlist.shortlist_vocab import StaticShortlist


def test_quantized_embedding_slice_is_bit_exact() -> None:
    ensure_metal_gpu_arch()
    mx.random.seed(42)
    base_emb = nn.Embedding(2048, 256)
    base_emb.weight = base_emb.weight.astype(mx.bfloat16)
    q_emb = nn.QuantizedEmbedding.from_embedding(base_emb, group_size=64, bits=8)
    mx.eval(q_emb.weight, q_emb.scales, q_emb.biases)

    static_ids = tuple(range(0, 256, 2))  # 128 non-contiguous rows
    static_sl = StaticShortlist(token_ids=static_ids, vocab_size=2048)
    static_head = slice_quantized_embedding(q_emb, static_ids)

    prompt_ids = [0, 2, 1001, 1505, 1001, 2047]
    prompt_head, dyn = build_prompt_sub_head(q_emb, static_sl, static_head, prompt_ids)
    assert dyn.extra_prompt_ids == (1001, 1505, 2047)
    assert prompt_head.vocab_size == 131

    x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
    full_logits = q_emb.as_linear(x)[0, 0, :]
    sub_logits = project_shortlist_logits(x, prompt_head)[0, 0, :]
    expected_sub = full_logits[prompt_head.token_ids_mx]
    mx.eval(sub_logits, expected_sub)

    max_diff = float(mx.max(mx.abs(sub_logits - expected_sub)).item())
    assert max_diff == 0.0


def test_predict_shortlist_draft_uses_logz_head_to_suppress_oov() -> None:
    ensure_metal_gpu_arch()
    mx.random.seed(11)
    base_emb = nn.Embedding(512, 128)
    base_emb.weight = base_emb.weight.astype(mx.bfloat16)
    q_emb = nn.QuantizedEmbedding.from_embedding(base_emb, group_size=64, bits=8)
    mx.eval(q_emb.weight, q_emb.scales, q_emb.biases)

    x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    full_pred = predict_full_head_draft(x, q_emb)

    # Deliberately exclude full_pred.token_id to simulate an OOV draft step
    shortlist_ids = tuple(i for i in range(64) if i != full_pred.token_id)
    sub_head = slice_quantized_embedding(q_emb, shortlist_ids)

    uncalibrated = predict_shortlist_draft(x, sub_head, logz_head=None)
    # Build a calibrator that outputs a much higher full logZ (+6.0 above shortlist logZ)
    high_logz_cal = LogZCalibrator(
        weight=np.zeros((128,), dtype=np.float32),
        norm_weight=0.0,
        bias=uncalibrated.predicted_logz + 6.0,
        ridge_lambda=1.0,
        n_samples=1,
    )
    calibrated = predict_shortlist_draft(
        x,
        sub_head,
        logz_head=compile_mlx_logz_head(high_logz_cal),
        calibrator_hidden=x,
        deadband_margin=0.0,
    )
    guarded = predict_shortlist_draft(
        x,
        sub_head,
        logz_head=compile_mlx_logz_head(high_logz_cal),
        calibrator_hidden=x,
        deadband_margin=2.0,
    )

    assert calibrated.token_id == uncalibrated.token_id
    assert calibrated.calibrated_prob < uncalibrated.shortlist_prob * 0.01
    assert calibrated.calibrated_prob < guarded.calibrated_prob < uncalibrated.shortlist_prob * 0.05
    assert 0.0 < calibrated.calibrated_prob <= 1.0
