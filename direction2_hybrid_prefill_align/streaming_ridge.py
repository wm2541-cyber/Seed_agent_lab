"""Pure functional shared-Gram streaming Ridge solver, SVD low-rank factorization, and sink-orthogonal projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SharedGramAccumulator:
    """Immutable sufficient statistics sharing one X^T X across multiple alignment target heads."""

    input_dim: int
    xtx: np.ndarray  # (D + 1, D + 1) float64
    xty_by_key: tuple[tuple[str, np.ndarray], ...]  # each (D + 1, out_dim) float64
    n_samples: int


@dataclass(frozen=True)
class AffineLinearMap:
    """Immutable affine map y = x @ weight + bias."""

    weight: np.ndarray  # (in_dim, out_dim) float32
    bias: np.ndarray  # (out_dim,) float32
    ridge_lambda: float
    n_samples: int


@dataclass(frozen=True)
class LowRankAffineMap:
    """Immutable rank-r factorized affine map y = (x @ left) @ right + bias."""

    left: np.ndarray  # (in_dim, rank) float32
    right: np.ndarray  # (rank, out_dim) float32
    bias: np.ndarray  # (out_dim,) float32
    rank: int
    retained_energy: float


def _augment_ones(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input (N, D), got shape {arr.shape}")
    ones = np.ones((arr.shape[0], 1), dtype=np.float64)
    return np.concatenate([arr, ones], axis=1)


def init_shared_gram(input_dim: int, target_dims: Mapping[str, int]) -> SharedGramAccumulator:
    """Purely create an empty shared-Gram streaming Ridge accumulator."""
    feat_dim = input_dim + 1
    xtx = np.zeros((feat_dim, feat_dim), dtype=np.float64)
    xtx.setflags(write=False)
    items: list[tuple[str, np.ndarray]] = []
    for key in sorted(target_dims.keys()):
        arr = np.zeros((feat_dim, int(target_dims[key])), dtype=np.float64)
        arr.setflags(write=False)
        items.append((str(key), arr))
    return SharedGramAccumulator(
        input_dim=input_dim,
        xtx=xtx,
        xty_by_key=tuple(items),
        n_samples=0,
    )


def update_shared_gram(
    acc: SharedGramAccumulator,
    x_batch: np.ndarray,
    targets_by_key: Mapping[str, np.ndarray],
) -> SharedGramAccumulator:
    """Purely return a new accumulator updated with one batch of shared inputs `x_batch` and named targets."""
    x_aug = _augment_ones(x_batch)
    n_batch = int(x_aug.shape[0])
    if x_aug.shape[1] != acc.input_dim + 1:
        raise ValueError(f"Input dim mismatch: expected {acc.input_dim}, got {x_aug.shape[1] - 1}")
    if n_batch == 0:
        return acc

    new_xtx = acc.xtx + (x_aug.T @ x_aug)
    new_xtx.setflags(write=False)

    new_items: list[tuple[str, np.ndarray]] = []
    for key, prev_xty in acc.xty_by_key:
        if key not in targets_by_key:
            raise KeyError(f"Missing target key {key!r} in batch update")
        y_arr = np.asarray(targets_by_key[key], dtype=np.float64)
        if y_arr.ndim != 2 or y_arr.shape[0] != n_batch or y_arr.shape[1] != prev_xty.shape[1]:
            raise ValueError(
                f"Target {key!r} shape mismatch: expected ({n_batch}, {prev_xty.shape[1]}), got {y_arr.shape}"
            )
        updated = prev_xty + (x_aug.T @ y_arr)
        updated.setflags(write=False)
        new_items.append((key, updated))

    return SharedGramAccumulator(
        input_dim=acc.input_dim,
        xtx=new_xtx,
        xty_by_key=tuple(new_items),
        n_samples=acc.n_samples + n_batch,
    )


def solve_shared_gram_ridge(
    acc: SharedGramAccumulator,
    ridge_lambda: float = 100.0,
) -> dict[str, AffineLinearMap]:
    """Purely solve all targets simultaneously using ONE regularized Gram factorization (unpenalized bias)."""
    if acc.n_samples <= 0:
        raise ValueError("Cannot solve Ridge from an empty accumulator")
    if ridge_lambda < 0.0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")

    feat_dim = acc.input_dim + 1
    reg = np.eye(feat_dim, dtype=np.float64) * float(ridge_lambda)
    reg[-1, -1] = 0.0  # never penalize intercept
    gram = acc.xtx + reg

    # Cholesky solve for symmetric positive definite regularized Gram matrix (fallback to np.linalg.solve if lambda==0)
    use_chol = ridge_lambda > 0.0
    chol_l = np.linalg.cholesky(gram) if use_chol else None

    solved: dict[str, AffineLinearMap] = {}
    for key, xty in acc.xty_by_key:
        if use_chol and chol_l is not None:
            z = np.linalg.solve(chol_l, xty)
            coef = np.linalg.solve(chol_l.T, z)
        else:
            coef = np.linalg.solve(gram, xty)
        w = np.asarray(coef[:-1, :], dtype=np.float32).copy()
        b = np.asarray(coef[-1, :], dtype=np.float32).copy()
        w.setflags(write=False)
        b.setflags(write=False)
        solved[key] = AffineLinearMap(
            weight=w,
            bias=b,
            ridge_lambda=float(ridge_lambda),
            n_samples=acc.n_samples,
        )
    return solved


def factorize_affine_svd(affine: AffineLinearMap, rank: int) -> LowRankAffineMap:
    """Purely truncate an AffineLinearMap's weight matrix to rank `r` via SVD: W ≈ left @ right."""
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    w64 = affine.weight.astype(np.float64)
    u, s, vt = np.linalg.svd(w64, full_matrices=False)
    r = min(int(rank), int(s.shape[0]))
    total_e = float(np.sum(s * s))
    kept_e = float(np.sum(s[:r] * s[:r]))
    ratio = kept_e / total_e if total_e > 1e-18 else 1.0

    sqrt_s = np.sqrt(s[:r])
    left = np.asarray(u[:, :r] * sqrt_s[None, :], dtype=np.float32).copy()
    right = np.asarray(sqrt_s[:, None] * vt[:r, :], dtype=np.float32).copy()
    bias = np.asarray(affine.bias, dtype=np.float32).copy()
    left.setflags(write=False)
    right.setflags(write=False)
    bias.setflags(write=False)
    return LowRankAffineMap(
        left=left,
        right=right,
        bias=bias,
        rank=r,
        retained_energy=ratio,
    )


def project_out_head_sinks(
    values_flat: np.ndarray,
    mean_v_heads: np.ndarray,
) -> np.ndarray:
    """Purely project out each head's Rank-1 Sink direction v_hat from `(..., Hv * Dv)` vectors.

    Maps each head vector `v_h -> (I - v_hat_h v_hat_h^T) v_h`, isolating the exact subspace that writes
    exclusively into the orthogonal dynamic GDN content `ax` while leaving Rank-1 sink `b` untouched.
    """
    mv = np.asarray(mean_v_heads, dtype=np.float64)  # (Hv, Dv)
    if mv.ndim != 2:
        raise ValueError(f"Expected mean_v_heads (Hv, Dv), got shape {mv.shape}")
    hv, dv = mv.shape
    norms = np.linalg.norm(mv, axis=-1, keepdims=True)
    v_hat = np.where(norms > 1e-12, mv / np.maximum(norms, 1e-12), 0.0)  # (Hv, Dv)

    arr = np.asarray(values_flat, dtype=np.float64)
    if arr.shape[-1] != hv * dv:
        raise ValueError(f"Last dim {arr.shape[-1]} != Hv*Dv ({hv * dv})")
    reshaped = arr.reshape(*arr.shape[:-1], hv, dv)
    proj_coeff = np.sum(reshaped * v_hat, axis=-1, keepdims=True)
    perp = reshaped - proj_coeff * v_hat
    return perp.reshape(arr.shape).astype(values_flat.dtype, copy=False)


def apply_affine_numpy(affine: AffineLinearMap, x: np.ndarray) -> np.ndarray:
    """Purely evaluate `x @ weight + bias` in float32."""
    return (np.asarray(x, dtype=np.float32) @ affine.weight) + affine.bias


def apply_low_rank_numpy(lr_map: LowRankAffineMap, x: np.ndarray) -> np.ndarray:
    """Purely evaluate `(x @ left) @ right + bias` in float32."""
    return ((np.asarray(x, dtype=np.float32) @ lr_map.left) @ lr_map.right) + lr_map.bias
