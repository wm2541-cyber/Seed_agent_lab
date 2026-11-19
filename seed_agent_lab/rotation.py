"""Pure ridge-regression linear rotation probes to align shallow hidden states to final layer space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearRotationMap:
    """Affine alignment map x -> [x, 1] @ weight_with_bias in hidden space."""

    weight_with_bias: np.ndarray  # shape (D + 1, D)
    ridge_lambda: float


def fit_linear_rotation(
    source_hidden: np.ndarray,
    target_hidden: np.ndarray,
    ridge_lambda: float = 10.0,
) -> LinearRotationMap:
    """Fit a closed-form ridge-regularized affine map from shallow to final hidden space."""
    src = np.asarray(source_hidden, dtype=np.float64)
    tgt = np.asarray(target_hidden, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2:
        raise ValueError(f"Expected matching 2D matrices, got {src.shape} and {tgt.shape}")

    n_samples, dim = src.shape
    if n_samples == 0:
        raise ValueError("Cannot fit rotation on empty sample set")

    ones = np.ones((n_samples, 1), dtype=np.float64)
    design = np.concatenate([src, ones], axis=1)  # (N, D + 1)

    gram = design.T @ design
    reg = np.eye(dim + 1, dtype=np.float64) * ridge_lambda
    reg[-1, -1] = 1e-6  # Do not shrink bias heavily.
    rhs = design.T @ tgt
    weights = np.linalg.solve(gram + reg, rhs)
    return LinearRotationMap(weight_with_bias=weights, ridge_lambda=ridge_lambda)


def apply_linear_rotation(
    rotation: LinearRotationMap,
    source_hidden: np.ndarray,
) -> np.ndarray:
    """Apply a fitted affine alignment map without mutating inputs."""
    src = np.asarray(source_hidden, dtype=np.float64)
    ones = np.ones((src.shape[0], 1), dtype=np.float64)
    design = np.concatenate([src, ones], axis=1)
    return design @ rotation.weight_with_bias
