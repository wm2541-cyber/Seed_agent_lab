"""Fast MLX aligned shallow-prefill execution engine supporting 4-layer Attn KV + 11-layer GDN FreezeB/Ax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_next import normalize_qk
import numpy as np

from direction2_hybrid_prefill_align.kv_gdn_aligner import (
    AttnKVMap,
    GDNHeadValueMap,
    HybridPrefillAligner,
    truncate_attn_kv_map,
)
from direction2_hybrid_prefill_align.streaming_ridge import AffineLinearMap
from seed_agent_lab.seed_ops import attention_write_only, gdn_extract_v_and_write


@dataclass(frozen=True)
class CompiledAttnKVLayer:
    """Immutable MLX GPU weights for one aligned deep Full-Attention KV write layer."""

    layer_idx: int
    k_target_is_post_norm: bool
    is_low_rank: bool
    k_w1: mx.array  # (D, kv_dim) or (D, r)
    k_w2: mx.array | None  # None or (r, kv_dim)
    k_b: mx.array  # (kv_dim,)
    v_w1: mx.array
    v_w2: mx.array | None
    v_b: mx.array


@dataclass(frozen=True)
class CompiledGDNLayer:
    """Immutable MLX GPU weights for one aligned deep GDN layer (FreezeB-Ax or FullV + optional K)."""

    layer_idx: int
    freeze_b: bool
    v_left: mx.array  # (Hv, D, r) bfloat16
    v_right: mx.array  # (Hv, r, Dv) bfloat16
    v_bias: mx.array  # (Hv, Dv) bfloat16
    unit_v_bar: mx.array  # (Hv, Dv) bfloat16
    k_weight: mx.array | None  # (D, key_dim) bfloat16
    k_bias: mx.array | None  # (key_dim,) bfloat16


@dataclass(frozen=True)
class CompiledHybridAligner:
    """Immutable MLX GPU representation of a HybridPrefillAligner ready for low-latency prefill."""

    anchor_layer: int
    attn_by_layer: Mapping[int, CompiledAttnKVLayer]
    gdn_ax_by_layer: Mapping[int, CompiledGDNLayer]
    gdn_fullv_by_layer: Mapping[int, CompiledGDNLayer]
    mean_v_by_layer: Mapping[int, mx.array]
    mean_a_by_layer: Mapping[int, mx.array]


@dataclass(frozen=True)
class AlignedPrefillConfig:
    """Immutable configuration controlling which alignment heads are active during shallow prefill."""

    align_attn_kv: bool = True
    gdn_mode: str = "none"  # "none", "freeze_b_fit_ax", "fit_full_v"
    align_gdn_k: bool = False
    anchor_mean_sink_b: bool = False


def rms_normalize_hidden(hidden: mx.array, eps: float = 1e-6) -> mx.array:
    """Pure parameter-free RMS normalization of `(B, S, D)` hidden states for scale-invariant alignment features."""
    h32 = hidden.astype(mx.float32)
    inv_rms = mx.rsqrt(mx.mean(h32 * h32, axis=-1, keepdims=True) + eps)
    return (h32 * inv_rms).astype(hidden.dtype)


def _compile_attn_layer(kv_map: AttnKVMap, rank: int | None) -> CompiledAttnKVLayer:
    if rank is not None and rank > 0:
        klr, vlr = truncate_attn_kv_map(kv_map, rank=rank)
        k_w1 = mx.array(klr.left, dtype=mx.bfloat16)
        k_w2 = mx.array(klr.right, dtype=mx.bfloat16)
        k_b = mx.array(klr.bias, dtype=mx.bfloat16)
        v_w1 = mx.array(vlr.left, dtype=mx.bfloat16)
        v_w2 = mx.array(vlr.right, dtype=mx.bfloat16)
        v_b = mx.array(vlr.bias, dtype=mx.bfloat16)
        mx.eval(k_w1, k_w2, k_b, v_w1, v_w2, v_b)
        return CompiledAttnKVLayer(
            layer_idx=kv_map.layer_idx,
            k_target_is_post_norm=kv_map.k_target_is_post_norm,
            is_low_rank=True,
            k_w1=k_w1,
            k_w2=k_w2,
            k_b=k_b,
            v_w1=v_w1,
            v_w2=v_w2,
            v_b=v_b,
        )
    k_w1 = mx.array(kv_map.k_map.weight, dtype=mx.bfloat16)
    k_b = mx.array(kv_map.k_map.bias, dtype=mx.bfloat16)
    v_w1 = mx.array(kv_map.v_map.weight, dtype=mx.bfloat16)
    v_b = mx.array(kv_map.v_map.bias, dtype=mx.bfloat16)
    mx.eval(k_w1, k_b, v_w1, v_b)
    return CompiledAttnKVLayer(
        layer_idx=kv_map.layer_idx,
        k_target_is_post_norm=kv_map.k_target_is_post_norm,
        is_low_rank=False,
        k_w1=k_w1,
        k_w2=None,
        k_b=k_b,
        v_w1=v_w1,
        v_w2=None,
        v_b=v_b,
    )


def _compile_gdn_layer(
    gmap: GDNHeadValueMap,
    k_map: AffineLinearMap | None,
) -> CompiledGDNLayer:
    v_left = mx.array(gmap.left, dtype=mx.bfloat16)
    v_right = mx.array(gmap.right, dtype=mx.bfloat16)
    v_bias = mx.array(gmap.bias, dtype=mx.bfloat16)
    unit_v = mx.array(gmap.unit_v_bar, dtype=mx.bfloat16)
    k_w = mx.array(k_map.weight, dtype=mx.bfloat16) if k_map is not None else None
    k_b = mx.array(k_map.bias, dtype=mx.bfloat16) if k_map is not None else None
    to_eval = [v_left, v_right, v_bias, unit_v] + ([k_w, k_b] if k_w is not None and k_b is not None else [])
    mx.eval(*to_eval)
    return CompiledGDNLayer(
        layer_idx=gmap.layer_idx,
        freeze_b=(gmap.mode == "freeze_b_fit_ax"),
        v_left=v_left,
        v_right=v_right,
        v_bias=v_bias,
        unit_v_bar=unit_v,
        k_weight=k_w,
        k_bias=k_b,
    )


def compile_hybrid_aligner(
    aligner: HybridPrefillAligner,
    attn_rank: int | None = None,
) -> CompiledHybridAligner:
    """Purely lift a NumPy HybridPrefillAligner to MLX bfloat16 GPU arrays (with optional Attn SVD rank)."""
    attn_by = {m.layer_idx: _compile_attn_layer(m, rank=attn_rank) for m in aligner.attn_kv_maps}
    k_dict = dict(aligner.gdn_k_maps)
    gdn_ax_by = {m.layer_idx: _compile_gdn_layer(m, k_dict.get(m.layer_idx)) for m in aligner.gdn_ax_maps}
    gdn_fullv_by = {m.layer_idx: _compile_gdn_layer(m, k_dict.get(m.layer_idx)) for m in aligner.gdn_fullv_maps}
    mv_by = {idx: mx.array(arr, dtype=mx.float32) for idx, arr in aligner.mean_v_by_layer}
    ma_by = {idx: mx.array(arr, dtype=mx.float32) for idx, arr in aligner.mean_a_by_layer}
    if mv_by:
        mx.eval(*mv_by.values(), *ma_by.values())
    return CompiledHybridAligner(
        anchor_layer=aligner.anchor_layer,
        attn_by_layer=attn_by,
        gdn_ax_by_layer=gdn_ax_by,
        gdn_fullv_by_layer=gdn_fullv_by,
        mean_v_by_layer=mv_by,
        mean_a_by_layer=ma_by,
    )


def attention_aligned_write_only(
    layer_module,
    anchor_normed: mx.array,
    compiled_kv: CompiledAttnKVLayer,
    cache,
) -> None:
    """Write aligned deep Full-Attention K/V directly from shallow `rms_normalize_hidden(h_k)` into `cache`."""
    attn = layer_module.self_attn
    B, L, _ = anchor_normed.shape
    if compiled_kv.is_low_rank and compiled_kv.k_w2 is not None and compiled_kv.v_w2 is not None:
        keys = ((anchor_normed @ compiled_kv.k_w1) @ compiled_kv.k_w2) + compiled_kv.k_b
        values = ((anchor_normed @ compiled_kv.v_w1) @ compiled_kv.v_w2) + compiled_kv.v_b
    else:
        keys = (anchor_normed @ compiled_kv.k_w1) + compiled_kv.k_b
        values = (anchor_normed @ compiled_kv.v_w1) + compiled_kv.v_b

    keys = keys.reshape(B, L, attn.num_key_value_heads, -1)
    if not compiled_kv.k_target_is_post_norm:
        keys = attn.k_norm(keys)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    keys = attn.rope(keys, offset=cache.offset)
    cache.update_and_fetch(keys, values)


def gdn_aligned_write_only(
    layer_module,
    x_hidden: mx.array,
    anchor_normed: mx.array,
    compiled_gdn: CompiledGDNLayer,
    align_gdn_k: bool,
    cache,
) -> None:
    """Write aligned deep GDN state into `cache`: freezes Rank-1 sink `b` when `freeze_b=True`, aligning only `ax`."""
    gdn = layer_module.linear_attn
    x_normed = layer_module.input_layernorm(x_hidden)
    B, S, _ = x_normed.shape
    qkv = gdn.in_proj_qkv(x_normed)
    b_gate = gdn.in_proj_b(x_normed)
    a_gate = gdn.in_proj_a(x_normed)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=x_normed.dtype)

    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = gdn.conv_kernel_size - 1
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

    conv_out = nn.silu(gdn.conv1d(conv_input))
    q, k, v_sh = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]

    # Fast 2-step per-head low-rank projection: single GEMM (N, D) @ (D, Hv*r) then batched (Hv, N, r) @ (Hv, r, Dv)
    hv, d_in, r_dim = compiled_gdn.v_left.shape
    flat_x = anchor_normed.reshape(B * S, d_in)
    w_left_2d = compiled_gdn.v_left.transpose(1, 0, 2).reshape(d_in, hv * r_dim)
    mid = (flat_x @ w_left_2d).reshape(B * S, hv, r_dim).transpose(1, 0, 2)  # (Hv, N, r)
    v_pred = (mid @ compiled_gdn.v_right).transpose(1, 0, 2).reshape(B, S, hv, gdn.head_v_dim)
    v_pred = v_pred + compiled_gdn.v_bias[None, None, :, :]

    if compiled_gdn.freeze_b:
        # Preserve shallow Rank-1 sink component along v_hat; replace ONLY orthogonal dynamic component (ax)!
        v_hat = compiled_gdn.unit_v_bar[None, None, :, :]  # (1, 1, Hv, Dv)
        sink_proj = mx.sum(v_sh * v_hat, axis=-1, keepdims=True) * v_hat
        # Re-orthogonalize v_pred against v_hat to prevent bfloat16 sink drift
        v_pred_perp = v_pred - mx.sum(v_pred * v_hat, axis=-1, keepdims=True) * v_hat
        v_use = sink_proj + v_pred_perp
    else:
        v_use = v_pred

    if align_gdn_k and compiled_gdn.k_weight is not None and compiled_gdn.k_bias is not None:
        k_pred = ((anchor_normed @ compiled_gdn.k_weight) + compiled_gdn.k_bias).reshape(
            B, S, gdn.num_k_heads, gdn.head_k_dim
        )
        k_use = k_pred
    else:
        k_use = k

    state = cache[1] if cache is not None else None
    q_n, k_n = normalize_qk(q, k_use, inv_scale=gdn.head_k_dim**-0.5, eps=1e-6)
    _, state = gated_delta_update(
        q_n, k_n, v_use, a_gate, b_gate, gdn.A_log, gdn.dt_bias, state, mask=None, use_kernel=True
    )
    if cache is not None:
        cache[1] = state
        cache.advance(S)


def _anchor_deep_gdn_sinks_to_mean(
    model,
    cache: Sequence,
    compiled: CompiledHybridAligner,
    start_layer: int,
) -> None:
    """Replace deep GDN states' Rank-1 Sink component `b` with training-set `mean_b`, leaving `ax` exact."""
    for idx in range(start_layer, len(model.layers)):
        layer = model.layers[idx]
        if not layer.is_linear or idx not in compiled.mean_v_by_layer:
            continue
        st = cache[idx][1]
        if st is None:
            continue
        orig_dtype = st.dtype
        st_f32 = st[0].astype(mx.float32)  # (Hv, Dv, Dk)
        mv = compiled.mean_v_by_layer[idx]  # (Hv, Dv)
        ma = compiled.mean_a_by_layer[idx]  # (Hv, Dk)
        v_hat = mv / mx.maximum(mx.sqrt(mx.sum(mv * mv, axis=-1, keepdims=True)), 1e-12)
        # S_perp = S - outer(v_hat, v_hat^T S); S_new = S_perp + outer(v_hat, ma)
        cur_a = mx.einsum("hd,hdk->hk", v_hat, st_f32)
        delta_b = mx.einsum("hd,hk->hdk", v_hat, ma - cur_a)
        cache[idx][1] = (st_f32 + delta_b)[None, ...].astype(orig_dtype)


def run_aligned_shallow_prefill(
    model,
    chunks_and_depths: Sequence[tuple[Sequence[int], int]],
    compiled: CompiledHybridAligner | None = None,
    config: AlignedPrefillConfig | None = None,
) -> list:
    """Run sequential prefill chunks with optional Upgraded Direction 2 alignment on layers >= anchor_k."""
    text_model = model.language_model.model
    cache = model.make_cache()
    num_layers = len(model.layers)
    cfg = config or AlignedPrefillConfig(align_attn_kv=False, gdn_mode="none")
    any_shallow = False

    for chunk_ids, anchor_k in chunks_and_depths:
        if not chunk_ids:
            continue
        hidden = text_model.embed_tokens(mx.array([list(chunk_ids)], dtype=mx.int32))
        fa_mask = create_attention_mask(hidden, cache[text_model.fa_idx])
        ssm_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx])
        k_clamped = max(1, min(int(anchor_k), num_layers))

        for idx in range(k_clamped):
            layer = model.layers[idx]
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=cache[idx])

        if k_clamped < num_layers:
            any_shallow = True
            use_aligned = compiled is not None and k_clamped == compiled.anchor_layer
            anchor_normed = rms_normalize_hidden(hidden) if use_aligned else None

            for idx in range(k_clamped, num_layers):
                layer = model.layers[idx]
                if not layer.is_linear:
                    if (
                        use_aligned
                        and compiled is not None
                        and cfg.align_attn_kv
                        and anchor_normed is not None
                        and idx in compiled.attn_by_layer
                    ):
                        attention_aligned_write_only(layer, anchor_normed, compiled.attn_by_layer[idx], cache[idx])
                    else:
                        attention_write_only(layer, hidden, cache[idx])
                else:
                    gdn_table = (
                        compiled.gdn_ax_by_layer
                        if (use_aligned and compiled is not None and cfg.gdn_mode == "freeze_b_fit_ax")
                        else compiled.gdn_fullv_by_layer
                        if (use_aligned and compiled is not None and cfg.gdn_mode == "fit_full_v")
                        else None
                    )
                    if gdn_table is not None and anchor_normed is not None and idx in gdn_table:
                        gdn_aligned_write_only(
                            layer,
                            hidden,
                            anchor_normed,
                            gdn_table[idx],
                            align_gdn_k=cfg.align_gdn_k,
                            cache=cache[idx],
                        )
                    else:
                        _ = gdn_extract_v_and_write(layer, hidden, cache[idx])

        mx.eval(
            [
                x
                for c in cache
                for x in (c.cache if isinstance(c, ArraysCache) else (c.keys, c.values))
                if x is not None
            ]
        )
        del hidden, fa_mask, ssm_mask
        mx.clear_cache()

    if any_shallow and compiled is not None and cfg.anchor_mean_sink_b:
        _anchor_deep_gdn_sinks_to_mean(model, cache, compiled, start_layer=compiled.anchor_layer + 1)
        mx.eval([c[1] for l, c in zip(model.layers, cache) if l.is_linear and c[1] is not None])

    return cache
