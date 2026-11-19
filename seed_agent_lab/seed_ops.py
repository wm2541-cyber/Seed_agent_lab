"""Pure SEED operations: b + ax State Sink decomposition, rank-R compression, and MLX write-only paths."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk


@dataclass(frozen=True)
class DecomposedState:
    """Immutable b + ax representation of one (Dv, Dk) recurrent state head."""

    unit_v_bar: np.ndarray  # (Dv,)
    sink_coeff: np.ndarray  # (Dk,)
    content: np.ndarray  # (Dv, Dk)

    def reconstruct(self) -> np.ndarray:
        """Reconstruct full (Dv, Dk) matrix b + ax = outer(unit_v_bar, sink_coeff) + content."""
        return np.outer(self.unit_v_bar, self.sink_coeff) + self.content


def decompose_state_sink(state_dv_dk: np.ndarray, mean_v: np.ndarray) -> DecomposedState:
    """Split a (Dv, Dk) GatedDeltaNet head state into rank-1 sink b and orthogonal content ax."""
    mat = np.asarray(state_dv_dk, dtype=np.float64)
    v_vec = np.asarray(mean_v, dtype=np.float64)
    norm_v = float(np.linalg.norm(v_vec))
    if norm_v <= 1e-12:
        unit_v = np.zeros_like(v_vec)
        unit_v[0] = 1.0
    else:
        unit_v = v_vec / norm_v

    sink_coeff = mat.T @ unit_v  # (Dk,)
    sink_mat = np.outer(unit_v, sink_coeff)
    content = mat - sink_mat
    return DecomposedState(unit_v_bar=unit_v, sink_coeff=sink_coeff, content=content)


def compress_matrix_svd(mat: np.ndarray, rank: int) -> np.ndarray:
    """Return best rank-R Frobenius approximation of a 2D matrix via SVD."""
    if rank <= 0:
        return np.zeros_like(mat, dtype=np.float64)
    arr = np.asarray(mat, dtype=np.float64)
    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    r = min(rank, len(s))
    return (u[:, :r] * s[:r]) @ vt[:r, :]


def compress_seed_state(state_dv_dk: np.ndarray, mean_v: np.ndarray, rank: int) -> np.ndarray:
    """Keep exact sink b and truncate only orthogonal content ax to rank R."""
    dec = decompose_state_sink(state_dv_dk, mean_v)
    compressed_content = compress_matrix_svd(dec.content, rank=rank)
    return np.outer(dec.unit_v_bar, dec.sink_coeff) + compressed_content


def gdn_extract_v_and_write(layer_module, x_hidden: mx.array, cache) -> mx.array:
    """Run GatedDeltaNet write path into cache and return post-conv values v (B, S, Hv, Dv)."""
    gdn = layer_module.linear_attn
    x_normed = layer_module.input_layernorm(x_hidden)
    B, S, _ = x_normed.shape
    qkv = gdn.in_proj_qkv(x_normed)
    b = gdn.in_proj_b(x_normed)
    a = gdn.in_proj_a(x_normed)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=x_normed.dtype)

    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = gdn.conv_kernel_size - 1
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

    conv_out = nn.silu(gdn.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]
    state = cache[1] if cache is not None else None
    q, k = normalize_qk(q, k, inv_scale=gdn.head_k_dim**-0.5, eps=1e-6)
    _, state = gated_delta_update(
        q, k, v, a, b, gdn.A_log, gdn.dt_bias, state, mask=None, use_kernel=True
    )
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    return v


def attention_write_only(layer_module, x_hidden: mx.array, cache) -> None:
    """Run only K/V projection, k_norm, RoPE, and cache append for a full-attention layer."""
    attn = layer_module.self_attn
    x_normed = layer_module.input_layernorm(x_hidden)
    B, L, _ = x_normed.shape
    keys = attn.k_proj(x_normed)
    values = attn.v_proj(x_normed)
    keys = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    keys = attn.rope(keys, offset=cache.offset)
    cache.update_and_fetch(keys, values)
