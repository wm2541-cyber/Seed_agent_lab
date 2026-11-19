"""Lightweight Layer-8 Draft Adapter modules and streaming Ridge accumulator in MLX/NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from seed_agent_lab.rotation import LinearRotationMap


@dataclass(frozen=True)
class StreamingRidgeAccumulator:
    """Immutable-style helper state for constant-memory Ridge normal equations."""

    gram: np.ndarray  # (D + 1, D + 1)
    rhs: np.ndarray  # (D + 1, D)
    count: int


def init_ridge_accumulator(dim: int) -> StreamingRidgeAccumulator:
    """Create empty Ridge normal-equation accumulators."""
    return StreamingRidgeAccumulator(
        gram=np.zeros((dim + 1, dim + 1), dtype=np.float64),
        rhs=np.zeros((dim + 1, dim), dtype=np.float64),
        count=0,
    )


def update_ridge_accumulator(
    acc: StreamingRidgeAccumulator,
    src_hidden: np.ndarray,
    tgt_hidden: np.ndarray,
) -> StreamingRidgeAccumulator:
    """Return updated Ridge accumulators from one trajectory batch without mutating inputs."""
    src = np.asarray(src_hidden, dtype=np.float64)
    tgt = np.asarray(tgt_hidden, dtype=np.float64)
    ones = np.ones((src.shape[0], 1), dtype=np.float64)
    design = np.concatenate([src, ones], axis=1)
    return StreamingRidgeAccumulator(
        gram=acc.gram + (design.T @ design),
        rhs=acc.rhs + (design.T @ tgt),
        count=acc.count + src.shape[0],
    )


def solve_ridge_accumulator(acc: StreamingRidgeAccumulator, ridge_lambda: float = 50.0) -> LinearRotationMap:
    """Solve accumulated Ridge system for W_k."""
    dim_plus_1 = acc.gram.shape[0]
    reg = np.eye(dim_plus_1, dtype=np.float64) * ridge_lambda
    reg[-1, -1] = 1e-6
    weights = np.linalg.solve(acc.gram + reg, acc.rhs)
    return LinearRotationMap(weight_with_bias=weights, ridge_lambda=ridge_lambda)


class LinearDraftAdapter(nn.Module):
    """Pure linear rotation+bias map (D -> D), warm-started from closed-form Ridge."""

    def __init__(self, hidden_dim: int, init_map: LinearRotationMap | None = None):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        if init_map is not None:
            w = init_map.weight_with_bias[:-1, :].T  # (D, D)
            b = init_map.weight_with_bias[-1, :]  # (D,)
            self.proj.weight = mx.array(w, dtype=mx.float32)
            self.proj.bias = mx.array(b, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class SwiGLUDraftAdapter(nn.Module):
    """Warm-started linear projection + tiny zero-init SwiGLU residual adapter."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 512,
        init_map: LinearRotationMap | None = None,
    ):
        super().__init__()
        self.base = nn.Linear(hidden_dim, hidden_dim, bias=True)
        if init_map is not None:
            w = init_map.weight_with_bias[:-1, :].T
            b = init_map.weight_with_bias[-1, :]
            self.base.weight = mx.array(w, dtype=mx.float32)
            self.base.bias = mx.array(b, dtype=mx.float32)

        self.gate_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.down_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        # Zero-init down_proj so at step 0 the adapter equals exact Ridge W_k.
        self.down_proj.weight = mx.zeros((hidden_dim, bottleneck_dim), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        delta = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return base_out + delta
