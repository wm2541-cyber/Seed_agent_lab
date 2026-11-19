"""Pure functional streaming Ridge calibrator predicting full-vocab logsumexp (logZ) from hidden states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StreamingLogZAccumulator:
    """Immutable sufficient statistics for streaming Ridge regression of full-vocab logZ."""

    feature_dim: int
    xtx: np.ndarray  # (F, F) float64
    xty: np.ndarray  # (F,) float64
    n_samples: int


@dataclass(frozen=True)
class LogZCalibrator:
    """Immutable scalar calibrator: logZ_hat(h) = h @ weight + norm_weight * ||h||_2 + bias."""

    weight: np.ndarray  # (D,) float32
    norm_weight: float
    bias: float
    ridge_lambda: float
    n_samples: int


@dataclass(frozen=True)
class LogZMetrics:
    """Immutable evaluation metrics for logZ prediction and OOV confidence calibration."""

    n_samples: int
    mae: float
    rmse: float
    r2: float
    mean_bias: float


def extract_logz_features(rotated_hidden: np.ndarray) -> np.ndarray:
    """Purely map rotated hidden vectors (N, D) -> [h, ||h||_2, 1.0] of shape (N, D + 2) float64.

    Including ||h||_2 captures the radial convex growth of logsumexp(E h) under sign flips/scaling.
    """
    h64 = np.asarray(rotated_hidden, dtype=np.float64)
    if h64.ndim == 1:
        h64 = h64[None, :]
    if h64.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {h64.shape}")
    l2_norm = np.linalg.norm(h64, axis=-1, keepdims=True)
    ones = np.ones((h64.shape[0], 1), dtype=np.float64)
    return np.concatenate([h64, l2_norm, ones], axis=-1)


def init_logz_accumulator(hidden_dim: int) -> StreamingLogZAccumulator:
    """Create a fresh immutable streaming accumulator for `hidden_dim` inputs."""
    feat_dim = hidden_dim + 2
    return StreamingLogZAccumulator(
        feature_dim=feat_dim,
        xtx=np.zeros((feat_dim, feat_dim), dtype=np.float64),
        xty=np.zeros((feat_dim,), dtype=np.float64),
        n_samples=0,
    )


def update_logz_accumulator(
    acc: StreamingLogZAccumulator,
    rotated_hidden: np.ndarray,
    full_logz: np.ndarray,
) -> StreamingLogZAccumulator:
    """Purely return a new accumulator updated with a batch of `(rotated_hidden, full_logz)`."""
    feats = extract_logz_features(rotated_hidden)
    targets = np.asarray(full_logz, dtype=np.float64).reshape(-1)
    if feats.shape[0] != targets.shape[0]:
        raise ValueError(f"Batch mismatch: {feats.shape[0]} hidden vectors vs {targets.shape[0]} targets")
    if feats.shape[1] != acc.feature_dim:
        raise ValueError(f"Feature dim mismatch: expected {acc.feature_dim}, got {feats.shape[1]}")
    if feats.shape[0] == 0:
        return acc
    return StreamingLogZAccumulator(
        feature_dim=acc.feature_dim,
        xtx=acc.xtx + (feats.T @ feats),
        xty=acc.xty + (feats.T @ targets),
        n_samples=acc.n_samples + int(feats.shape[0]),
    )


def solve_logz_calibrator(
    acc: StreamingLogZAccumulator,
    ridge_lambda: float = 10.0,
) -> LogZCalibrator:
    """Pure closed-form Ridge solve leaving the scalar intercept unpenalized."""
    if acc.n_samples == 0:
        raise ValueError("Cannot solve LogZCalibrator from an empty accumulator")
    if ridge_lambda < 0.0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")

    reg = np.eye(acc.feature_dim, dtype=np.float64) * float(ridge_lambda)
    # Do not shrink the scalar bias (last column) or norm coefficient aggressively:
    # keep intercept unpenalized, small stability floor on norm weight.
    reg[-1, -1] = 0.0
    reg[-2, -2] = min(float(ridge_lambda), 1e-2)
    coef = np.linalg.solve(acc.xtx + reg, acc.xty)

    hidden_dim = acc.feature_dim - 2
    weight = np.asarray(coef[:hidden_dim], dtype=np.float32).copy()
    weight.setflags(write=False)
    return LogZCalibrator(
        weight=weight,
        norm_weight=float(coef[-2]),
        bias=float(coef[-1]),
        ridge_lambda=float(ridge_lambda),
        n_samples=acc.n_samples,
    )


def predict_logz_numpy(
    calibrator: LogZCalibrator,
    rotated_hidden: np.ndarray,
) -> np.ndarray:
    """Purely predict full-vocab logsumexp from rotated hidden vectors."""
    h64 = np.asarray(rotated_hidden, dtype=np.float64)
    squeeze = h64.ndim == 1
    if squeeze:
        h64 = h64[None, :]
    w64 = calibrator.weight.astype(np.float64)
    l2_norm = np.linalg.norm(h64, axis=-1)
    pred = (h64 @ w64) + calibrator.norm_weight * l2_norm + calibrator.bias
    return pred[0] if squeeze else pred


def calibrated_max_prob_numpy(
    shortlist_logits: np.ndarray,
    predicted_full_logz: np.ndarray | float,
    deadband_margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Purely compute shortlist argmax index and calibrated max probability in (0, 1].

    Uses exact mathematical lower bound with optional OOV-guard deadband:
      logZ_eff = max(logsumexp(shortlist_logits), predicted_full_logz - deadband_margin).
    """
    if deadband_margin < 0.0:
        raise ValueError(f"deadband_margin must be non-negative, got {deadband_margin}")
    logits = np.asarray(shortlist_logits, dtype=np.float64)
    max_idx = np.argmax(logits, axis=-1)
    max_val = np.max(logits, axis=-1)
    shortlist_logz = max_val + np.log(np.sum(np.exp(logits - np.expand_dims(max_val, axis=-1)), axis=-1))
    shifted_pred = np.asarray(predicted_full_logz, dtype=np.float64) - float(deadband_margin)
    effective_logz = np.maximum(shortlist_logz, shifted_pred)
    prob = np.exp(max_val - effective_logz)
    return max_idx, prob


def evaluate_logz_calibrator(
    calibrator: LogZCalibrator,
    rotated_hidden: np.ndarray,
    true_full_logz: np.ndarray,
) -> LogZMetrics:
    """Purely compute regression quality metrics (MAE, RMSE, R^2, bias) on held-out targets."""
    preds = np.asarray(predict_logz_numpy(calibrator, rotated_hidden), dtype=np.float64).reshape(-1)
    truth = np.asarray(true_full_logz, dtype=np.float64).reshape(-1)
    if preds.shape != truth.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs truth {truth.shape}")
    if truth.size == 0:
        raise ValueError("Cannot evaluate on empty target array")
    diff = preds - truth
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ss_res = float(np.sum(diff * diff))
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    return LogZMetrics(
        n_samples=int(truth.size),
        mae=mae,
        rmse=rmse,
        r2=r2,
        mean_bias=float(np.mean(diff)),
    )


def save_logz_calibrator(path: Path, calibrator: LogZCalibrator) -> Path:
    """Save a LogZCalibrator cleanly to `.npz`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weight=np.asarray(calibrator.weight, dtype=np.float32),
        norm_weight=np.asarray([calibrator.norm_weight], dtype=np.float64),
        bias=np.asarray([calibrator.bias], dtype=np.float64),
        ridge_lambda=np.asarray([calibrator.ridge_lambda], dtype=np.float64),
        n_samples=np.asarray([calibrator.n_samples], dtype=np.int64),
    )
    return path


def load_logz_calibrator(path: Path) -> LogZCalibrator:
    """Load an immutable LogZCalibrator from `.npz`."""
    data = np.load(path)
    weight = np.asarray(data["weight"], dtype=np.float32).copy()
    weight.setflags(write=False)
    return LogZCalibrator(
        weight=weight,
        norm_weight=float(data["norm_weight"][0]),
        bias=float(data["bias"][0]),
        ridge_lambda=float(data["ridge_lambda"][0]),
        n_samples=int(data["n_samples"][0]),
    )
