"""Exact row-sliced QuantizedEmbedding LM head and MLX logZ-calibrated draft prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import numpy as np

from direction1_phase_shortlist.logz_calibrator import LogZCalibrator
from direction1_phase_shortlist.shortlist_vocab import DynamicShortlist, StaticShortlist, build_dynamic_shortlist


@dataclass(frozen=True)
class QuantizedSubHead:
    """Immutable row-sliced quantized LM head (bit-exact subset of QuantizedEmbedding.as_linear)."""

    token_ids_tuple: tuple[int, ...]
    token_ids_mx: mx.array  # (K,) int32
    weight: mx.array  # (K, D_packed) uint32
    scales: mx.array  # (K, n_groups) bfloat16
    biases: mx.array  # (K, n_groups) bfloat16
    group_size: int
    bits: int

    @property
    def vocab_size(self) -> int:
        return len(self.token_ids_tuple)


@dataclass(frozen=True)
class MLXLogZHead:
    """Immutable MLX GPU weights for scalar full-vocab logZ estimation."""

    weight_f32: mx.array  # (D,) float32
    norm_weight: float
    bias: float


@dataclass(frozen=True)
class DraftPrediction:
    """Immutable result of one Layer-k draft prediction step."""

    token_id: int
    calibrated_prob: float
    shortlist_prob: float
    predicted_logz: float


def compile_mlx_logz_head(calibrator: LogZCalibrator) -> MLXLogZHead:
    """Purely lift a NumPy LogZCalibrator onto MLX float32 arrays."""
    weight_f32 = mx.array(calibrator.weight, dtype=mx.float32)
    mx.eval(weight_f32)
    return MLXLogZHead(
        weight_f32=weight_f32,
        norm_weight=calibrator.norm_weight,
        bias=calibrator.bias,
    )


def slice_quantized_embedding(embed_tokens, token_ids: Sequence[int]) -> QuantizedSubHead:
    """Purely extract an exact quantized sub-head for `token_ids` (per-row quantization along hidden dim)."""
    ids_tuple = tuple(int(t) for t in token_ids)
    if not ids_tuple:
        raise ValueError("token_ids must be non-empty")
    ids_mx = mx.array(list(ids_tuple), dtype=mx.int32)
    w_sub = embed_tokens.weight[ids_mx]
    s_sub = embed_tokens.scales[ids_mx]
    b_sub = embed_tokens.biases[ids_mx]
    mx.eval(ids_mx, w_sub, s_sub, b_sub)
    return QuantizedSubHead(
        token_ids_tuple=ids_tuple,
        token_ids_mx=ids_mx,
        weight=w_sub,
        scales=s_sub,
        biases=b_sub,
        group_size=int(embed_tokens.group_size),
        bits=int(embed_tokens.bits),
    )


def build_prompt_sub_head(
    embed_tokens,
    static_shortlist: StaticShortlist,
    static_sub_head: QuantizedSubHead,
    prompt_ids: Sequence[int],
    static_token_set: frozenset[int] | None = None,
) -> tuple[QuantizedSubHead, DynamicShortlist]:
    """Purely build a contiguous prompt-conditioned QuantizedSubHead by appending unique unseen prompt rows."""
    dynamic = build_dynamic_shortlist(static_shortlist, prompt_ids, static_token_set=static_token_set)
    if not dynamic.extra_prompt_ids:
        return static_sub_head, dynamic

    extra_ids_mx = mx.array(list(dynamic.extra_prompt_ids), dtype=mx.int32)
    extra_w = embed_tokens.weight[extra_ids_mx]
    extra_s = embed_tokens.scales[extra_ids_mx]
    extra_b = embed_tokens.biases[extra_ids_mx]

    merged_ids_mx = mx.concatenate([static_sub_head.token_ids_mx, extra_ids_mx], axis=0)
    merged_w = mx.concatenate([static_sub_head.weight, extra_w], axis=0)
    merged_s = mx.concatenate([static_sub_head.scales, extra_s], axis=0)
    merged_b = mx.concatenate([static_sub_head.biases, extra_b], axis=0)
    mx.eval(merged_ids_mx, merged_w, merged_s, merged_b)

    return (
        QuantizedSubHead(
            token_ids_tuple=dynamic.all_ids,
            token_ids_mx=merged_ids_mx,
            weight=merged_w,
            scales=merged_s,
            biases=merged_b,
            group_size=static_sub_head.group_size,
            bits=static_sub_head.bits,
        ),
        dynamic,
    )


def project_shortlist_logits(rotated_hidden: mx.array, sub_head: QuantizedSubHead) -> mx.array:
    """Compute exact shortlist logits via MLX quantized_matmul (bit-identical to full embed_tokens.as_linear)."""
    return mx.quantized_matmul(
        rotated_hidden,
        sub_head.weight,
        scales=sub_head.scales,
        biases=sub_head.biases,
        transpose=True,
        group_size=sub_head.group_size,
        bits=sub_head.bits,
    )


def predict_full_head_draft(
    rotated_hidden: mx.array,
    embed_tokens,
) -> DraftPrediction:
    """Compute full-vocabulary argmax token and exact softmax confidence."""
    logits = embed_tokens.as_linear(rotated_hidden)[0, -1, :]
    logits_f32 = logits.astype(mx.float32)
    tok = mx.argmax(logits, axis=-1)
    logz = mx.logsumexp(logits_f32, axis=-1)
    max_logit = mx.max(logits_f32, axis=-1)
    max_prob = mx.exp(max_logit - logz)
    mx.eval(tok, max_prob, logz)
    prob_val = float(max_prob.item())
    return DraftPrediction(
        token_id=int(tok.item()),
        calibrated_prob=prob_val,
        shortlist_prob=prob_val,
        predicted_logz=float(logz.item()),
    )


def predict_shortlist_draft(
    rotated_hidden: mx.array,
    sub_head: QuantizedSubHead,
    logz_head: MLXLogZHead | None = None,
    calibrator_hidden: mx.array | None = None,
    deadband_margin: float = 0.0,
) -> DraftPrediction:
    """Compute shortlist argmax token and logZ-calibrated max probability (prevents OOV false confidence)."""
    logits = project_shortlist_logits(rotated_hidden, sub_head)[0, -1, :]
    logits_f32 = logits.astype(mx.float32)
    local_idx = mx.argmax(logits, axis=-1)
    max_logit = mx.max(logits_f32, axis=-1)
    shortlist_logz = mx.logsumexp(logits_f32, axis=-1)
    shortlist_prob = mx.exp(max_logit - shortlist_logz)

    if logz_head is not None:
        cal_src = calibrator_hidden if calibrator_hidden is not None else rotated_hidden
        h_f32 = cal_src[0, -1, :].astype(mx.float32)
        h_norm = mx.sqrt(mx.sum(h_f32 * h_f32))
        pred_logz = mx.sum(h_f32 * logz_head.weight_f32) + (logz_head.norm_weight * h_norm) + logz_head.bias
        eff_logz = mx.maximum(shortlist_logz, pred_logz - float(deadband_margin))
        cal_prob = mx.exp(max_logit - eff_logz)
        mx.eval(local_idx, cal_prob, shortlist_prob, pred_logz)
        return DraftPrediction(
            token_id=sub_head.token_ids_tuple[int(local_idx.item())],
            calibrated_prob=float(cal_prob.item()),
            shortlist_prob=float(shortlist_prob.item()),
            predicted_logz=float(pred_logz.item()),
        )

    mx.eval(local_idx, shortlist_prob, shortlist_logz)
    s_prob = float(shortlist_prob.item())
    return DraftPrediction(
        token_id=sub_head.token_ids_tuple[int(local_idx.item())],
        calibrated_prob=s_prob,
        shortlist_prob=s_prob,
        predicted_logz=float(shortlist_logz.item()),
    )
